"""
main.py
FastAPI app — entry point for the whole system.
"""

import asyncio
import io
import json
import logging
import os
import time
from pathlib import Path
import wave
from groq import Groq
import numpy as np
import redis.asyncio as redis
import torch
import webrtcvad
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import base64
import re
from fastapi.staticfiles import StaticFiles


from config import (
    TTS_PROVIDER,
    SAMPLE_RATE, FRAME_BYTES, SILENCE_FRAMES, VAD_AGGRESSIVENESS, MAX_BUFFER_BYTES,
    MAX_HISTORY_TURNS,
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    RAG_METHOD,
    LLM_PROVIDER,
    STT_PROVIDER,
    MAX_TOKENS_SUMMARY,
    INTERVIEW_DURATION_SECONDS,
    POCKET_TTS_URL, POCKET_TTS_VOICE,
    INTERVIEW_WARNING_SECONDS,
)
from database import init_db, check_interview_status, register_new_user, archive_interview
from vectorstore import load_knowledge_base
from rag_factory import run_turn
from observability import track_turn, start_session_run, end_session_run
from llm_provider import llm_chat, get_fast_model

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
http_client = httpx.AsyncClient(timeout=30.0)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

app = FastAPI(title="AI Interviewer", version="2.0")

redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
    decode_responses=True,
)

session_tasks: dict = {}
ai_speaking_sessions: set = set()


vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)


@app.on_event("startup")
async def startup():
    log.info("=" * 60)
    log.info(f"AI Interviewer starting")
    log.info(f"  RAG_METHOD:   {RAG_METHOD}")
    log.info(f"  LLM_PROVIDER: {LLM_PROVIDER}")
    log.info(f"  TTS_PROVIDER: {TTS_PROVIDER}")
    log.info("=" * 60)
    await init_db()
    load_knowledge_base("./docs")
    try:
       async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(POCKET_TTS_URL, data={"text": "Hello", "voice_url": POCKET_TTS_VOICE})
       log.info("[TTS] Pocket TTS warmed up")
    except Exception as e:
      log.warning(f"[TTS] warmup failed: {e}")
    log.info("Ready.")


@app.get("/health")
async def health():
    status = {"rag_method": RAG_METHOD, "llm_provider": LLM_PROVIDER}
    try:
        await redis_client.ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"error: {e}"
    return JSONResponse(content=status)


@app.get("/")
async def serve_frontend():
    html_path = Path("static/index.html")
    if not html_path.exists():
        return HTMLResponse("<h1>Frontend not found</h1>")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


def _whisper_transcribe(audio_bytes: bytes) -> str:
    if not audio_bytes:
        return ""

    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(audio_int16) == 0:
        return ""

    # Energy gate BEFORE the API call — don't pay to transcribe silence
    avg_energy = np.abs(audio_int16).mean()
    if avg_energy < 300:
        log.info(f"[STT] Skipping low-energy audio (energy={avg_energy:.0f})")
        return ""

    # Groq needs a real audio file, so wrap raw PCM into a WAV in memory
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)      # mono
        wf.setsampwidth(2)      # 16-bit = 2 bytes
        wf.setframerate(16000)  # 16 kHz
        wf.writeframes(audio_bytes)
    buf.seek(0)
    buf.name = "audio.wav"      # the SDK uses the extension

    # ── API call (only this part needs the try/except) ──
    try:
        result = groq_client.audio.transcriptions.create(
            file=buf,
            model="whisper-large-v3-turbo",
            language="en",
            temperature=0.0,
            response_format="verbose_json",  # verbose_json gives segments
        )
    except Exception as e:
        log.error(f"[STT] Groq transcription failed: {e}")
        return ""

    text = (result.text or "").strip()
    if not text:
        return ""

    # ── Hallucination defense (runs after we safely have text) ──
    segments = getattr(result, "segments", None)
    if segments:
        def _no_speech(s):
            return s.get("no_speech_prob", 0) if isinstance(s, dict) else getattr(s, "no_speech_prob", 0)
        avg_no_speech = sum(_no_speech(s) for s in segments) / len(segments)
        if avg_no_speech > 0.6:
            log.info(f"[STT] Hallucination rejected (no_speech_prob={avg_no_speech:.2f}): '{text}'")
            return ""

    text_lower = text.lower().strip(".!?, ")

    if "interview about artificial" in text_lower:
        log.info(f"[STT] Rejected echoed initial_prompt: '{text}'")
        return ""

    HALLUCINATION_PHRASES = {
        "thank you for watching",
        "subscribe", "please subscribe", "subscribe to my channel",
        "like and subscribe","thank you"
    }
    if text_lower in HALLUCINATION_PHRASES:
        log.info(f"[STT] Hallucination phrase rejected: '{text}'")
        return ""

    if len(text) < 3:
        return ""

    return text


async def transcribe(audio_bytes: bytes) -> str:
    if not audio_bytes:
        return ""
    # All STT runs through Groq's hosted Whisper (fast). Kept in a thread
    # because the Groq SDK call is blocking and our pipeline is async.
    return await asyncio.to_thread(_whisper_transcribe, audio_bytes)


# ═════════════════════════════════════════════════════════════════
#  SEMANTIC ENDPOINT DETECTION (Technique 4)
# ═════════════════════════════════════════════════════════════════
def _is_thought_complete(text: str) -> bool:
    """
    Decide whether the user's utterance sounds like a FINISHED thought or a
    mid-sentence pause. This lets us end the turn fast when they're clearly
    done, but wait longer when they're obviously still mid-sentence — instead
    of a single fixed silence timer that always trades off cutting people off
    against feeling laggy. Rule-based, so it adds zero latency.

    Returns True  -> thought looks complete  -> respond quickly
    Returns False -> thought looks unfinished -> wait longer for continuation
    """
    t = text.strip().lower()
    if not t:
        return False

    # Words that, when a sentence ends on them, signal the speaker isn't done:
    # conjunctions, articles, prepositions, pronouns, auxiliaries, fillers.
    DANGLING_WORDS = {
        "and", "but", "or", "so", "because", "however", "although", "though",
        "while", "since", "if", "when", "as", "which", "who",
        "the", "a", "an", "my", "our", "their", "his", "her", "its",
        "to", "of", "for", "in", "on", "at", "with", "from", "by", "about",
        "i", "we", "they", "he", "she", "you",
        "is", "are", "was", "were", "will", "would", "can", "could", "should",
        "like", "really", "very", "just", "also", "then", "actually",
        "um", "uh", "er", "basically", "kind", "sort", "such" ,"ehh" ,"mm"
    }

    stripped = t.rstrip(".!?,;: ")
    words = stripped.split()
    last_word = words[-1] if words else ""

    # 1. Dangling connective/filler word -> clearly mid-sentence
    if last_word in DANGLING_WORDS:
        return False

    # 2. Ends with sentence-ending punctuation -> looks complete
    if t.endswith((".", "?", "!")):
        return True

    # 3. Trailing comma/semicolon/dash -> mid-list, not done
    if t.endswith((",", ";", ":", "-")):
        return False

    # 4. Default: treat as complete so we don't over-hold normal speech.
    #    (The safety-flush + merge logic still catches the rare miss.)
    return True


#TTS
async def _auto_release_speaking_lock(session_id: str, duration_seconds: float):
    await asyncio.sleep(duration_seconds)
    if session_id in ai_speaking_sessions:
        ai_speaking_sessions.discard(session_id)
        log.info(f"[STATE] {session_id}: TTS auto-released after {duration_seconds:.1f}s")


async def send_response_for_speech(websocket: WebSocket, session_id: str, text: str):
    """
    STREAMING TTS. The LLM has already produced the full `text` (LangGraph/RAG
    is untouched). We stream Pocket TTS's WAV bytes to the browser as they
    render — first audio in ~90ms instead of waiting ~5s for the whole clip.

    Protocol over the WebSocket:
      AUDIO_START (json)  -> browser prepares a fresh streaming player
      <binary frames>     -> raw WAV bytes (header in the first frame)
      AUDIO_END   (json)  -> browser knows no more chunks are coming
    Barge-in: if the session leaves ai_speaking_sessions mid-stream, we stop
    forwarding chunks immediately.
    """
    if not text or not text.strip():
        return

    ai_speaking_sessions.add(session_id)

    try:
        await websocket.send_json({"type": "AUDIO_START"})

        async with http_client.stream(
            "POST",
            POCKET_TTS_URL,
            data={"text": text, "voice_url": POCKET_TTS_VOICE},
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size=4096):
                # Barge-in: session was discarded -> stop streaming now
                if session_id not in ai_speaking_sessions:
                    log.info("[TTS] Barge-in mid-stream, stopping audio")
                    break
                if chunk:
                    await websocket.send_bytes(chunk)

        await websocket.send_json({"type": "AUDIO_END"})

    except Exception as e:
        log.error(f"[TTS] Pocket TTS stream failed: {e}")
        # fallback: browser speaks the text so the interview never stalls
        try:
            await websocket.send_json({"type": "SPEAK", "text": text})
        except Exception:
            pass
        return

    # failsafe lock release — the real release is the browser's TTS_DONE
    # once playback finishes. This only catches a stuck browser.
    word_count = len(text.split())
    estimated_seconds = (word_count / 2.5) + 12.0
    asyncio.create_task(_auto_release_speaking_lock(session_id, estimated_seconds))

#USED FOR LISTENING
async def vad_receiver_loop(
    websocket: WebSocket, session_id: str,
    audio_queue: asyncio.Queue, stop_event: asyncio.Event,
):
    MIN_SPEECH_FRAMES = 10
    CONFIRM_SPEECH_FRAMES = 8
    BARGE_IN_CONFIRM_FRAMES = 15

    speech_buffer    = bytearray()
    raw_frame_buf    = bytearray()
    silence_count    = 0
    speech_count     = 0
    speech_started   = False
    listening_sent   = False
    verifying_barge_in = False
    barge_in_text = ""

    while not stop_event.is_set():
        try:
            message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except (WebSocketDisconnect, RuntimeError):
            break

        if message.get("type") == "websocket.disconnect":
            break

        if "text" in message:
            try:
                data = json.loads(message["text"])
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "")

            if msg_type == "INTERRUPT":
                if session_id in ai_speaking_sessions:
                    verifying_barge_in = True
                    barge_in_text = data.get("spoken_so_far", "")
                    ai_speaking_sessions.discard(session_id)
                    log.info("[BARGE-IN] Frontend signaled interrupt. Listening to verify...")
                continue
            elif msg_type == "TTS_DONE":
                ai_speaking_sessions.discard(session_id)
                verifying_barge_in = False

            continue

        if "bytes" not in message:
            continue

        if session_id in ai_speaking_sessions:
            raw_frame_buf.clear()
            speech_buffer.clear()
            silence_count   = 0
            speech_count    = 0
            speech_started  = False
            listening_sent  = False
            session_tasks.setdefault(session_id, {})["is_speaking"] = False
            continue

        raw_frame_buf.extend(message["bytes"])

        while len(raw_frame_buf) >= FRAME_BYTES:
            frame         = bytes(raw_frame_buf[:FRAME_BYTES])
            raw_frame_buf = raw_frame_buf[FRAME_BYTES:]

            frame_np = np.frombuffer(frame, dtype=np.int16)
            energy = np.abs(frame_np).mean()

            if energy < 200:
                is_speech = False
            else:
                try:
                    is_speech = vad.is_speech(frame, SAMPLE_RATE)
                except Exception:
                    is_speech = False

            if is_speech:
                silence_count  = 0
                speech_count  += 1
                speech_buffer.extend(frame)

                if speech_count == CONFIRM_SPEECH_FRAMES and not speech_started:
                    speech_started = True
                    session_tasks.setdefault(session_id, {})["is_speaking"] = True
                    if not listening_sent:
                        try:
                            await websocket.send_json({"type": "STATUS", "message": "listening"})
                            listening_sent = True
                        except Exception:
                            pass

                if speech_count == BARGE_IN_CONFIRM_FRAMES and verifying_barge_in:
                    await audio_queue.put(("interrupt", barge_in_text))
                    verifying_barge_in = False
                    log.info(f"[BARGE-IN] Verified! Speech sustained for {BARGE_IN_CONFIRM_FRAMES} frames. AI stopped.")

            elif speech_count > 0:
                silence_count += 1
                speech_buffer.extend(frame)

                if silence_count >= SILENCE_FRAMES:
                    speech_started = False
                    session_tasks.setdefault(session_id, {})["is_speaking"] = False
                    verifying_barge_in = False

                    if speech_count >= MIN_SPEECH_FRAMES:
                        try:
                            await websocket.send_json({"type": "STATUS", "message": "transcribing"})
                        except Exception:
                            pass
                        await audio_queue.put(("audio", bytes(speech_buffer)))
                    else:
                        log.info(f"[VAD] Dropped short noise ({speech_count} frames)")
                        await audio_queue.put(("audio", b""))

                    speech_buffer.clear()
                    silence_count   = 0
                    speech_count    = 0
                    listening_sent  = False

            if len(speech_buffer) >= MAX_BUFFER_BYTES:
                speech_started = False
                session_tasks.setdefault(session_id, {})["is_speaking"] = False
                verifying_barge_in = False

                if speech_count >= MIN_SPEECH_FRAMES:
                    await audio_queue.put(("audio", bytes(speech_buffer)))
                else:
                    await audio_queue.put(("audio", b""))

                speech_buffer.clear()
                silence_count   = 0
                speech_count    = 0
                listening_sent  = False

async def turn_processor_loop(
    websocket: WebSocket, session_id: str,
    audio_queue: asyncio.Queue, stop_event: asyncio.Event,
):
    turn_number = 0
    while not stop_event.is_set():
        try:
            item = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        kind, payload = item

        # ── SAFETY NET TO PREVENT SILENT CRASHES ──
        try:
            if kind == "interrupt":
                await handle_barge_in(websocket, session_id, payload)
            elif kind == "audio":
                turn_number += 1
                await process_turn(websocket, session_id, payload, stop_event, turn_number, audio_queue)
            elif kind == "partial":
                await process_partial_audio(websocket, session_id, payload)
        except Exception as e:
            log.error(f"[FATAL ERROR] The background processor crashed: {e}", exc_info=True)
            try:
                await websocket.send_json({"type": "STATUS", "message": "API Error - Check Terminal"})
            except Exception:
                pass

        audio_queue.task_done()

#HAVE TO WORK ON IT
async def process_partial_audio(
    websocket: WebSocket,
    session_id: str,
    audio_bytes: bytes,
):
    partial_text = await transcribe(audio_bytes)
    if not partial_text:
        log.info(f"[PARTIAL] No transcription from buffer (silence or noise)")
        return

    raw = await redis_client.get(session_id)
    if not raw:
        return
    state = json.loads(raw)

    existing = state.get("turn_buffer", "").strip()
    if existing:
        state["turn_buffer"] = f"{existing} {partial_text}"
    else:
        state["turn_buffer"] = partial_text

    await redis_client.set(session_id, json.dumps(state))
    log.info(f"[PARTIAL] Saved into turn_buffer: '{partial_text}'")

    try:
        await websocket.send_json({"type": "TRANSCRIPT", "text": f"{state['turn_buffer']} ..."})
    except Exception:
        pass


async def handle_barge_in(websocket: WebSocket, session_id: str, text_spoken: str):
    log.info(f"[BARGE-IN] {session_id}: user interrupted Sarah")

    try:
        await websocket.send_json({"type": "STOP_SPEAKING"})
    except Exception:
        pass

    tasks = session_tasks.get(session_id, {"summary_task": None})
    raw   = await redis_client.get(session_id)
    if not raw:
        return

    state = json.loads(raw)

    if tasks.get("summary_task") and not tasks["summary_task"].done():
        tasks["summary_task"].cancel()
        log.info("[BARGE-IN] Cancelled in-flight summary task")
    elif tasks.get("summary_task") and tasks["summary_task"].done():
        state["summary"] = state.get("previous_summary", state["summary"])
        log.info("[BARGE-IN] Rolled summary back to previous version")

    if state["history"] and state["history"][-1]["role"] == "assistant":
        state["history"][-1]["content"] = f"{text_spoken} ... [Interrupted by candidate]"

    await redis_client.set(session_id, json.dumps(state))


async def update_summary(session_id: str):
    raw = await redis_client.get(session_id)
    if not raw:
        return

    state = json.loads(raw)
    if len(state["history"]) <= MAX_HISTORY_TURNS:
        return

    state["previous_summary"] = state["summary"]
    await redis_client.set(session_id, json.dumps(state))

    recent_buffer = state["history"][-(MAX_HISTORY_TURNS * 2):]
    prompt = (
        f"Update the rolling interview summary in 2-3 sentences.\n"
        f"Previous summary: {state['summary']}\n"
        f"Recent turns: {json.dumps(recent_buffer)}\n"
        f"Include: topics covered, candidate's strength areas, any red flags."
    )

    try:
        text, _ = await llm_chat(
            messages=[{"role": "user", "content": prompt}],
            model=get_fast_model(),
            max_tokens=MAX_TOKENS_SUMMARY,
            temperature=0.3,
        )
        state["summary"] = text.strip()
        await redis_client.set(session_id, json.dumps(state))
        log.info(f"[SUMMARY] Updated for {session_id}: '{state['summary'][:80]}...'")
    except asyncio.CancelledError:
        log.info(f"[SUMMARY] Cancelled for {session_id}")
        raise
    except Exception as e:
        log.error(f"[SUMMARY] Failed: {e}")


async def process_turn(
    websocket: WebSocket, session_id: str,
    user_audio: bytes, stop_event: asyncio.Event,
    turn_number: int,
    audio_queue: asyncio.Queue,
):
    await websocket.send_json({"type": "STATUS", "message": "transcribing"})
    user_text = await transcribe(user_audio)

    raw = await redis_client.get(session_id)
    if not raw:
        return
    state = json.loads(raw)

    if user_text:
        existing_buffer = state.get("turn_buffer", "").strip()
        if existing_buffer:
            state["turn_buffer"] = f"{existing_buffer} {user_text}"
            log.info(f"[MERGE] Buffer now: '{state['turn_buffer'][:70]}...'")
        else:
            bargein = state.get("pending_user_text", "").strip()
            if bargein:
                state["turn_buffer"] = f"{bargein}, {user_text}"
                state["pending_user_text"] = ""
            else:
                state["turn_buffer"] = user_text

        await redis_client.set(session_id, json.dumps(state))
        await websocket.send_json({"type": "TRANSCRIPT", "text": state["turn_buffer"]})

    full_user_text = state.get("turn_buffer", "").strip()
    if not full_user_text:
        return

    is_speaking = session_tasks.get(session_id, {}).get("is_speaking", False)

    # ── SEMANTIC ENDPOINT CHECK (Technique 4) ──
    # Decide whether the utterance sounds finished. This shapes how patient
    # we are: an incomplete thought ("I worked at Google for...") should wait
    # for the continuation; a complete thought ("...for three years.") can be
    # answered right away.
    thought_complete = _is_thought_complete(full_user_text)

    if is_speaking or not audio_queue.empty():
        log.info(f"[HOLD] User still speaking. Holding: '{full_user_text[:50]}...'")
        # incomplete thought -> wait longer; complete -> short safety net
        grace = 2 if not thought_complete else 0.5
        _schedule_safety_flush(websocket, session_id, stop_event, turn_number, audio_queue, grace)
        return

    if not thought_complete:
        # Quiet right now, but the sentence trails off mid-thought — the user
        # is likely just pausing to think. Hold briefly and let them continue
        # instead of cutting in on an unfinished sentence.
        log.info(f"[ENDPOINT] Incomplete thought, waiting for continuation: '{full_user_text[:50]}...'")
        _schedule_safety_flush(websocket, session_id, stop_event, turn_number, audio_queue, 2)
        return

    # Quiet AND the thought sounds complete -> respond immediately
    log.info(f"[ENDPOINT] Complete thought, responding now: '{full_user_text[:50]}...'")
    await _commit_and_respond(websocket, session_id, state, full_user_text, turn_number)


def _schedule_safety_flush(websocket, session_id, stop_event, turn_number, audio_queue, grace=0.5):
    """
    Cancel any existing safety-flush task and schedule a new one.
    `grace` is set by the caller based on semantic completeness — a longer
    wait for an unfinished sentence, a shorter one for a complete thought.
    """
    slot = session_tasks.setdefault(session_id, {})
    old = slot.get("safety_flush")
    if old and not old.done():
        old.cancel()
    slot["safety_flush"] = asyncio.create_task(
        _safety_flush(websocket, session_id, stop_event, turn_number, audio_queue, grace)
    )


async def _safety_flush(websocket, session_id, stop_event, turn_number, audio_queue, grace=0.5):
    """
    Wait `grace` seconds. If still quiet (not speaking, queue empty) and
    a buffer is waiting, fire the LLM. Prevents a held buffer from being
    stranded when no further audio arrives to re-trigger process_turn.
    """
    try:
        await asyncio.sleep(grace)

        if stop_event.is_set():
            return

        is_speaking = session_tasks.get(session_id, {}).get("is_speaking", False)
        if is_speaking or not audio_queue.empty():
            return  # user resumed or more audio queued — let normal flow handle it

        raw = await redis_client.get(session_id)
        if not raw:
            return
        state = json.loads(raw)
        full_user_text = state.get("turn_buffer", "").strip()
        if not full_user_text:
            return  # buffer already flushed by normal flow

        log.info(f"[SAFETY] Flushing buffer after {grace}s quiet: '{full_user_text[:50]}...'")
        await _commit_and_respond(websocket, session_id, state, full_user_text, turn_number)

    except asyncio.CancelledError:
        # New audio arrived and rescheduled us — normal, do nothing
        raise
    except Exception as e:
        log.error(f"[SAFETY] flush error: {e}", exc_info=True)


async def _commit_and_respond(websocket, session_id, state, full_user_text, turn_number):
    """
    Commit the buffer to history, call the LLM, send + speak the reply.
    Shared by the normal flow and the safety-flush fallback.

    DOUBLE-FIRE GUARD: a synchronous is_responding flag set before any await
    so the normal path and the safety flush can't both call the LLM (which
    previously caused a duplicate call and a 429 rate limit).
    """
    slot = session_tasks.setdefault(session_id, {})
    if slot.get("is_responding"):
        log.info("[GUARD] Already responding — skipping duplicate LLM call")
        return
    slot["is_responding"] = True

    try:
        # Re-read state fresh, double-check buffer still present
        raw = await redis_client.get(session_id)
        if not raw:
            return
        state = json.loads(raw)
        full_user_text = state.get("turn_buffer", "").strip()
        if not full_user_text:
            return  # someone already committed this turn

        state["history"].append({"role": "user", "content": full_user_text})
        state["turn_buffer"] = ""
        await redis_client.set(session_id, json.dumps(state))

        await websocket.send_json({"type": "STATUS", "message": "thinking"})

        recent_history = state["history"][-(MAX_HISTORY_TURNS * 2):]

        accumulated = state.get("accumulated_seconds", 0)
        time_remaining = max(0, INTERVIEW_DURATION_SECONDS - accumulated)

        with track_turn(session_id, turn_number) as log_metrics:
            result = await run_turn(
                user_text=full_user_text,
                history=recent_history,
                summary=state.get("summary", ""),
                time_remaining=time_remaining,
            )
            log_metrics.metrics(
                tokens=result["tokens_used"],
                rag_used=result["rag_used"],
            )

        ai_text     = result["response"]
        tokens_used = result["tokens_used"]
        log.info(f"[TURN {turn_number}] reply='{ai_text[:60]}...'")

        state["tokens_total"] = state.get("tokens_total", 0) + tokens_used
        state["history"].append({"role": "assistant", "content": ai_text})
        await redis_client.set(session_id, json.dumps(state))

        await websocket.send_json({"type": "AI_REPLY", "text": ai_text})
        await send_response_for_speech(websocket, session_id, ai_text)

        if len(state["history"]) > MAX_HISTORY_TURNS:
            slot["summary_task"] = asyncio.create_task(update_summary(session_id))

    finally:
        slot["is_responding"] = False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, name: str = "", email: str = ""):
    if not email:
        await websocket.accept()
        await websocket.send_json({"type": "ERROR", "message": "Email required"})
        await websocket.close()
        return

    status, session_id = await check_interview_status(email)
    if status == "ALREADY_COMPLETED":
        await websocket.accept()
        await websocket.send_json({"type": "ERROR", "message": "You already completed this interview."})
        await websocket.close()
        return

    await websocket.accept()
    log.info(f"[WS] {email} connected")

    if session_id:
        if session_id in session_tasks:
            for key in ("summary_task", "timer_task", "llm_task", "safety_flush"):
                old_task = session_tasks[session_id].get(key)
                if old_task and not old_task.done():
                    old_task.cancel()
            session_tasks.pop(session_id, None)
        ai_speaking_sessions.discard(session_id)
        log.info(f"[STATE FLUSH] {session_id}: cleared ephemeral session state")

    is_resume = False
    if session_id:
        cached = await redis_client.get(session_id)
        if cached:
            state = json.loads(cached)
            is_resume = True
            log.info(f"[RESUME] {email} reconnected — restoring {len(state['history'])} turns")
        else:
            state = _new_state()
            await redis_client.set(session_id, json.dumps(state))
    else:
        session_id = await register_new_user(name or "Anonymous", email)
        state      = _new_state()
        await redis_client.set(session_id, json.dumps(state))

    defaults = _new_state()
    for k, v in defaults.items():
        if k not in state:
            state[k] = v
    await redis_client.set(session_id, json.dumps(state))

    session_tasks[session_id] = {"summary_task": None, "timer_task": None,
                                 "llm_task": None, "safety_flush": None,
                                 "is_speaking": False, "is_responding": False}

    start_session_run(session_id, email)

    if not state["history"]:
        opening = (f"Hi {name or 'there'}, I'm Sarah. Thanks for taking the time today. "
                   f"To start could you walk me through your background and what drew you to this role?")
        state["history"].append({"role": "assistant", "content": opening})
        await redis_client.set(session_id, json.dumps(state))
        await asyncio.sleep(0.2)
        await websocket.send_json({"type": "AI_REPLY", "text": opening})
        await send_response_for_speech(websocket, session_id, opening)

    elif is_resume:
        accumulated = state.get("accumulated_seconds", 0)
        remaining   = max(0, INTERVIEW_DURATION_SECONDS - accumulated)
        resume_msg  = f"Welcome back. You have {remaining // 60} minutes {remaining % 60} seconds remaining. Please continue your previous answer."
        await websocket.send_json({
            "type":      "RESUME",
            "message":   f"Resuming interview. {remaining // 60}m {remaining % 60}s left.",
            "remaining": remaining,
        })
        await asyncio.sleep(0.3)
        await websocket.send_json({"type": "AI_REPLY", "text": resume_msg})
        await send_response_for_speech(websocket, session_id, resume_msg)

    audio_queue = asyncio.Queue()
    stop_event  = asyncio.Event()

    timer_task = asyncio.create_task(
        interview_timer(websocket, session_id, stop_event)
    )
    session_tasks[session_id]["timer_task"] = timer_task

    try:
        results = await asyncio.gather(
            vad_receiver_loop(websocket, session_id, audio_queue, stop_event),
            turn_processor_loop(websocket, session_id, audio_queue, stop_event),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, WebSocketDisconnect):
                log.info(f"[WS] {email} disconnected — state preserved for resume")
            elif isinstance(r, Exception):
                log.error(f"[WS] Loop error for {email}: {r}")
    except WebSocketDisconnect:
        log.info(f"[WS] {email} disconnected — Redis state preserved for resume")
    except Exception as e:
        log.error(f"[WS] Crash: {e}", exc_info=True)
    finally:
        stop_event.set()
        for key in ("summary_task", "timer_task", "llm_task", "safety_flush"):
            task = session_tasks.get(session_id, {}).get(key)
            if task and not task.done():
                task.cancel()
        session_tasks.pop(session_id, None)
        ai_speaking_sessions.discard(session_id)
        try:
            from observability import end_session_run
            end_session_run(total_turns=0, total_tokens=0, completed=False)
        except Exception:
            pass


async def interview_timer(
    websocket: WebSocket,
    session_id: str,
    stop_event: asyncio.Event,
):
    while not stop_event.is_set():
        raw = await redis_client.get(session_id)
        if not raw:
            return

        state = json.loads(raw)
        accumulated = state.get("accumulated_seconds", 0)
        remaining   = INTERVIEW_DURATION_SECONDS - accumulated

        await websocket.send_json({
            "type":      "TIMER_TICK",
            "remaining": int(remaining),
            "elapsed":   int(accumulated),
            "total":     INTERVIEW_DURATION_SECONDS,
        })

        if not state.get("warning_sent") and 0 < remaining <= INTERVIEW_WARNING_SECONDS:
            state["warning_sent"] = True
            await redis_client.set(session_id, json.dumps(state))
            await websocket.send_json({
                "type":    "TIMER_WARNING",
                "message": f"{INTERVIEW_WARNING_SECONDS} seconds remaining",
            })
            log.info(f"[TIMER] {session_id}: 1 minute warning")

        if remaining <= 0:
            log.info(f"[TIMER] {session_id}: time expired")
            await end_interview_now(websocket, session_id)
            stop_event.set()
            return

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
            return
        except asyncio.TimeoutError:
            pass

        raw = await redis_client.get(session_id)
        if not raw:
            return
        state = json.loads(raw)
        state["accumulated_seconds"] = state.get("accumulated_seconds", 0) + 1
        await redis_client.set(session_id, json.dumps(state))


async def end_interview_now(websocket: WebSocket, session_id: str):
    raw = await redis_client.get(session_id)
    if not raw:
        return
    state = json.loads(raw)

    task = session_tasks.get(session_id, {}).get("summary_task")
    if task and not task.done():
        task.cancel()

    await archive_interview(session_id, state, RAG_METHOD, LLM_PROVIDER)
    await redis_client.delete(session_id)

    end_session_run(
        total_turns=len(state["history"]) // 2,
        total_tokens=state.get("tokens_total", 0),
        completed=True,
    )

    log.info(f"[TIMER] {session_id}: interview complete (timer expired)")

    try:
        await websocket.send_json({
            "type":    "INTERVIEW_DONE",
            "message": "Time is up. Thank you — the interview is complete.",
        })
        await websocket.close()
    except Exception:
        pass


def _new_state() -> dict:
    return {
        "history":             [],
        "summary":             "",
        "previous_summary":    "",
        "tokens_total":        0,
        "status":              "IN_PROGRESS",
        "accumulated_seconds": 0,
        "warning_sent":        False,
        "pending_user_text":   "",
        "last_user_msg_time":  0,
        "turn_buffer":         "",
    }


@app.get("/test-llm")
async def test_llm():
    """Quick endpoint to verify LLM connection and token"""
    try:
        from llm_provider import llm_chat, get_fast_model
        reply, _ = await llm_chat(
            messages=[{"role": "user", "content": "Say about: 'LLM !'"}],
            model=get_fast_model(),
            max_tokens=20,
            temperature=0.0
        )
        return {"status": "SUCCESS", "response": reply}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}