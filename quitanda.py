#!/usr/bin/env python3
"""
quitanda_agent.py — LiveKit voice agent (PT-PT)
Churrascaria Quitanda · Python 3.12 · Maio 2025
"""

from __future__ import annotations
import asyncio, logging, os, json, re, time as pytime, functools
from dataclasses import dataclass
from datetime import datetime as _dt, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions, WorkerType
from livekit.agents.llm import function_tool
from livekit.plugins import openai
from openai.types.beta.realtime.session import TurnDetection
from livekit.protocol.sip import TransferSIPParticipantRequest
from livekit import api

# ───────────────────────── Config base ─────────────────────────
load_dotenv(".env.local")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger("quitanda")

# Observabilidade leve
CALL_METRICS = {}
def mark(event: str):
    CALL_METRICS[event] = _dt.now(TZ)

# Cache de dados frequentemente acessados
SHOP_STATE_CACHE = {"data": None, "expires_at": None}
CACHE_TTL = 300  # segundos (5 minutos)

def get_cached_or_compute(cache_dict, compute_fn, *args, **kwargs):
    """Retorna dados do cache se válidos, ou executa função e atualiza cache."""
    now = _dt.now(TZ)
    if cache_dict["data"] and cache_dict["expires_at"] and now < cache_dict["expires_at"]:
        log.debug("Usando dados do cache (válido até %s)", cache_dict["expires_at"])
        return cache_dict["data"]
    
    # Computar novos dados
    result = compute_fn(*args, **kwargs)
    
    # Atualizar cache
    cache_dict["data"] = result
    cache_dict["expires_at"] = now + timedelta(seconds=CACHE_TTL)
    log.debug("Cache atualizado, válido até %s", cache_dict["expires_at"])
    
    return result

PT_PT_NUDGE_WORDS = (
    "autocarro frigorífico sumo óptimo estádio campeã "
    "recepção telemóvel pastel de nata comboio voçês "
    "sandes azeite chouriço pastelaria obrigado "
    "rapariga miúdo pá fixe vá lá pequeno-almoço "
    "frigorífico casa de banho esplanada talho pingo doce "
    "à noite tipo assim de facto montra eléctrico "
    "apanhar o comboio apetece-me elefante perceber "
    "desculpe empregado factura guardanapo freguesia "
    "bilhete de identidade concelho cacete centrais "
    "bom dia boa tarde boa noite estou de acordo "
    "pois claro como estás meu caro tudo bem? ora pois "
    "café bica imperiais fino ginjinha fado saudade "
    "bacalhau à brás pastéis de belém francesinha "
)
# Bloco fantasma: não tem significado para o cliente,
# mas ajuda o TTS/OpenAI a detectar o dialecto correcto.
PT_PT_NUDGE_BLOCK = f"""<nudge>
{PT_PT_NUDGE_WORDS}
Pronúncia portuguesa autêntica:
- 'e' final como 'ə' (schwa), nunca como 'i'
- 's' final como 'sh' (chiante)
- vogais nasais fechadas ('pão', 'bem')
- 'r' vibrante uvular (garganta), não como 'r' brasileiro
- vogais átonas muito reduzidas/fechadas
- entoação típica de Lisboa/Porto, não do Brasil
- 'lh' como em 'filho' (não como 'lio')
- 'ou' pronunciado como 'ô' fechado
- 't' e 'd' como dentais, não africadas
</nudge>"""

# Exemplos de diálogo PT-PT para o modelo
PT_PT_EXAMPLES = """
Exemplos de diálogos AUTÊNTICOS em Português de Portugal:

Cliente: Queria fazer uma reserva para amanhã.
Operador: Com certeza. Para quantas pessoas e a que horas desejas a reserva?

Cliente: Tenho uma encomenda para levantar.
Operador: Diz-me o teu nome, se faz favor, para que possa verificar a tua encomenda.

Cliente: Quanto tempo demora a preparar um frango assado?
Operador: O nosso frango assado demora cerca de 30 minutos a ser preparado.

Cliente: Gostava de encomendar para levar.
Operador: Claro! O que queres encomendar e a que horas vens buscar?

Cliente: Fecha a que horas?
Operador: Hoje fechamos às 21:30. Queres fazer uma encomenda para hoje?

Cliente: Isto vem com batatas?
Operador: Sim, vem com batatas fritas. Queres acrescentar mais alguma coisa?

Cliente: Qual é o prato do dia?
Operador: Hoje temos bacalhau à Brás como prato do dia. Acompanha com batatas e salada.

Cliente: Quanto custa o frango inteiro?
Operador: O frango inteiro custa 12 euros. Queres adicionar algum acompanhamento?

Cliente: Podem entregar em casa?
Operador: Não fazemos entregas, mas podes encomendar para vires buscar à hora que te for mais conveniente.

Traços OBRIGATÓRIOS do sotaque português europeu:
- Sempre "tu" em vez de "você" ("Queres" em vez de "Você quer")
- Frases curtas e diretas, vocabulário europeu
- "Percebo", "estou a ver", "já agora", "se faz favor", "com certeza"
- "Casa de banho" (não "banheiro"), "pequeno-almoço" (não "café da manhã")
- "Telemóvel" (não "celular"), "autocarro" (não "ônibus"), "comboio" (não "trem")
- "Miúdo", "pá", "fixe", "giro", "gajo", "tipo", "bué", "já está"
- Pronúncia fechada de vogais átonas, especialmente 'e' final
- 's' final pronunciado como 'sh' (chiante)
- 'r' pronunciado na garganta (uvular)
- Evitar expressões brasileiras: "a gente", "legal", "tudo bem?", "você"
"""

try:
    TZ = ZoneInfo("Europe/Lisbon")
except ZoneInfoNotFoundError:
    TZ = timezone.utc
log.info("Timezone forçado: %s", TZ)

# chave global com tudo o que vem do Make
BOOT_DATA: dict[str, Any] = {"menu": None, "hours": None, "shop_state": None}

MAKE_URL = os.getenv("MENU_WEBHOOK_URL")
TRANSFER_PHONE_NUMBER = os.getenv("TRANSFER_PHONE_NUMBER", "+351933792547")

# ───────────────────────── Horário oficial ─────────────────────
RAW_SCHEDULE = {
    "monday":    ["10:00–14:00", "17:30–21:30"],
    "tuesday":   [],
    "wednesday": ["17:30–21:30"],
    "thursday":  ["10:00–14:00", "17:30–21:30"],
    "friday":    ["10:00–14:00", "17:30–21:30"],
    "saturday":  ["10:00–14:30", "17:00–21:30"],
    "sunday":    ["10:00–14:30", "17:00–21:30"],
}

@dataclass(frozen=True)
class TimeSlot:
    start_minutes: int
    end_minutes: int

def hms_to_minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h*60 + m

def minutes_to_hms(m: int) -> str:
    return f"{m//60:02}:{m%60:02}"

def convert_schedule(raw) -> dict[str, list[TimeSlot]]:
    out = {}
    for d, slots in raw.items():
        out[d] = []
        for s in slots:
            a, b = s.split("–")
            out[d].append(TimeSlot(hms_to_minutes(a), hms_to_minutes(b)))
    return out
SCHEDULE = convert_schedule(RAW_SCHEDULE)

class ShopStatus(Enum):
    OPEN   = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"

# ───────────────────────── BOOT fetch (1 vez) ──────────────────
async def boot_fetch_menu() -> None:
    """Carrega dados do menu e estado da loja com resiliência a falhas."""
    try:
        # Criar sessão com timeout para evitar bloqueios
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as sess:
            try:
                async with asyncio.TaskGroup() as tg:  # Python 3.12
                    t_menu = tg.create_task(sess.get(MAKE_URL, timeout=8))
                    t_state = tg.create_task(compute_shop_state())
                
                resp = t_menu.result()  # result() já retorna o objeto resolvido
                
                # Verificar status HTTP para tratar erros de API
                if resp.status != 200:
                    log.error(f"Erro ao carregar menu: HTTP {resp.status}")
                    BOOT_DATA["menu"] = "Menu temporariamente indisponível"
                    BOOT_DATA["shop_state"] = t_state.result()
                    return
                
                try:
                    raw = await resp.json(content_type=None)
                    
                    src = raw.get("dynamic_variables", raw)
                    BOOT_DATA["menu"] = src.get("menu_items", "Menu indisponível")
                    BOOT_DATA["hours"] = src.get("hoursanddate", "Horários indisponíveis")
                    BOOT_DATA["shop_state"] = t_state.result()
                    
                    log.info("BOOT_DATA carregado — menu %d chars, hours %d chars",
                            len(str(BOOT_DATA['menu'])), len(str(BOOT_DATA['hours'])))
                except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                    log.error(f"Erro ao processar JSON do menu: {str(e)}")
                    BOOT_DATA["menu"] = "Erro ao carregar menu"
                    BOOT_DATA["hours"] = "Horários indisponíveis"
                    BOOT_DATA["shop_state"] = t_state.result()
            
            except asyncio.CancelledError:
                log.warning("Carregamento do menu cancelado")
                # Garantir que pelo menos o estado da loja está disponível
                BOOT_DATA["shop_state"] = await compute_shop_state()
                BOOT_DATA["menu"] = "Menu indisponível (timeout)"
                BOOT_DATA["hours"] = "Horários indisponíveis (timeout)"
    
    except Exception as e:
        log.error(f"Erro crítico ao carregar dados iniciais: {str(e)}")
        # Fornecer dados de fallback para permitir que o sistema funcione
        BOOT_DATA["menu"] = "Menu temporariamente indisponível. Por favor, tente mais tarde."
        BOOT_DATA["hours"] = "Horários indisponíveis. Por favor, pergunte ao funcionário."
        
        # Garantir que pelo menos o estado da loja está disponível
        try:
            BOOT_DATA["shop_state"] = await compute_shop_state()
        except Exception as se:
            log.error(f"Erro ao calcular estado da loja: {str(se)}")
            BOOT_DATA["shop_state"] = {"status": ShopStatus.UNKNOWN.value,
                                       "message": "Estado indisponível. Por favor, tente novamente."}

# ───────────────────────── Helpers horário ─────────────────────
def get_todays_slots(now: _dt) -> list[TimeSlot]:
    return SCHEDULE.get(now.strftime("%A").lower(), [])

def find_next_available_time(now: _dt) -> Optional[_dt]:
    mins = now.hour*60 + now.minute
    # hoje
    for s in get_todays_slots(now):
        if s.start_minutes > mins:
            return now.replace(hour=s.start_minutes//60,
                               minute=s.start_minutes%60,
                               second=0, microsecond=0)
    # próximos dias
    for i in range(1, 8):
        nd = now + timedelta(days=i)
        sl = get_todays_slots(nd)
        if sl:
            f = sl[0]
            return nd.replace(hour=f.start_minutes//60,
                              minute=f.start_minutes%60,
                              second=0, microsecond=0)
    return None

async def compute_shop_state() -> dict:
    # Tenta obter do cache primeiro
    # Como esta função é assíncrona, precisamos adaptar a chamada
    if SHOP_STATE_CACHE["data"] and SHOP_STATE_CACHE["expires_at"]:
        now = _dt.now(TZ)
        if now < SHOP_STATE_CACHE["expires_at"]:
            log.debug("Usando estado da loja do cache (válido até %s)", 
                      SHOP_STATE_CACHE["expires_at"])
            return SHOP_STATE_CACHE["data"]
    
    # Se não estiver em cache ou expirado, calcula novo estado
    now = _dt.now(TZ)
    mins = now.hour*60 + now.minute
    slots = get_todays_slots(now)
    readable = ", ".join(f"{minutes_to_hms(s.start_minutes)}-{minutes_to_hms(s.end_minutes)}"
                         for s in slots) or "Encerrado hoje"
    result = None
    
    for s in slots:
        if s.start_minutes <= mins < s.end_minutes:
            result = {"status": ShopStatus.OPEN.value,
                    "next_close": minutes_to_hms(s.end_minutes),
                    "today_readable_hours": readable}
            break

    if not result:
        nxt = find_next_available_time(now)
        msg = "Estamos fechados."
        nxt_str = None
        if nxt:
            nxt_str = nxt.strftime("%H:%M")
            msg = f"Estamos fechados agora. Próxima abertura: {nxt_str}"
        result = {"status": ShopStatus.CLOSED.value,
                "next_open_time": nxt_str,
                "today_readable_hours": readable,
                "message": msg}
    
    # Atualiza o cache
    SHOP_STATE_CACHE["data"] = result
    SHOP_STATE_CACHE["expires_at"] = now + timedelta(seconds=CACHE_TTL)
    log.debug("Cache de estado da loja atualizado, válido até %s", 
              SHOP_STATE_CACHE["expires_at"])
    
    return result

# ───────────────────────── Ferramentas LLM ─────────────────────
@function_tool
async def get_menu(hours_only: bool=False) -> dict:
    """Devolve menu/hours do BOOT_DATA (sem HTTP)."""
    data = {
        "menu_items": BOOT_DATA["menu"]  or "Menu indisponível",
        "hoursanddate": BOOT_DATA["hours"] or "Horários indisponíveis",
        "menu_options": {},   # <— podes parsear se necessário
    }
    return {"hoursanddate": data["hoursanddate"]} if hours_only else data

@function_tool
async def get_shop_state(date_string: str="") -> dict:
    """Estado já calculado no boot."""
    return BOOT_DATA["shop_state"] or {"status": ShopStatus.UNKNOWN.value}

# ─────────────────────────  Validate Pick-up  ───────────────────
RELATIVE_TIME_REGEX = re.compile(r"daqui\s+a\s+(\d+)\s*(minutos|mins|min|horas|hrs|h)")

@functools.cache
def _extract_time_part(segment: str) -> Optional[str]:
    """Apanha HH:MM; ignora intervalos HH:MM-HH:MM"""
    if "Horário:" in segment:
        time_part = segment.split("Horário:")[1].split()[0]
    else:
        time_part = segment.split()[0]
    if "-" in time_part:
        time_part = time_part.split("-")[0]
    return time_part if ":" in time_part else None

@function_tool
async def validate_pickup(pickup_time: str,
                          hoursanddate: str="",
                          current_datetime: str="") -> dict:
    try:
        ph, pm = map(int, pickup_time.split(":"))
        p_minutes = ph*60 + pm
    except ValueError:
        return {"valid": False, "reason": f"Formato inválido: {pickup_time}"}

    now = _dt.now(TZ)
    c_minutes = now.hour*60 + now.minute
    if p_minutes <= c_minutes:
        return {"valid": False, "reason": f"{pickup_time} já passou."}

    slots = get_todays_slots(now)
    if not slots:
        return {"valid": False, "reason": "Estamos encerrados hoje."}

    in_slot = any(s.start_minutes <= p_minutes < s.end_minutes for s in slots)
    if not in_slot:
        readable = ", ".join(f"{minutes_to_hms(s.start_minutes)}-{minutes_to_hms(s.end_minutes)}"
                             for s in slots)
        return {"valid": False, "reason": f"{pickup_time} fora do horário. Hoje: {readable}"}

    # Usar hoursanddate do BOOT_DATA se não for fornecido
    if not hoursanddate:
        hoursanddate = BOOT_DATA["hours"] or ""
        
    avail = []
    for seg in hoursanddate.split("|"):
        tp = _extract_time_part(seg)
        if tp and "Disponível:Sim" in seg:
            avail.append(tp)
    if avail and pickup_time not in avail:
        return {"valid": False,
                "reason": f"{pickup_time} indisponível. Possíveis: {', '.join(avail)}"}

    return {"valid": True}

@function_tool
async def validate_pickup_combined(pickup_time: str,
                                   current_datetime: str="",
                                   raw_expression: str=None) -> dict:
    """
    Validação robusta de horário de recolha com informações detalhadas.
    Unifica validação de horário e informações do estado da loja.
    """
    try:
        # Usa dados locais; chama validate_pickup para reaproveitar lógica
        hours = BOOT_DATA["hours"] or ""
        
        # Formato de hora inválido
        try:
            ph, pm = map(int, pickup_time.split(":"))
            p_minutes = ph*60 + pm
        except (ValueError, TypeError):
            return {
                "valid": False, 
                "reason": f"Formato inválido: {pickup_time}. Use o formato HH:MM.",
                "shop_status": BOOT_DATA["shop_state"].get("status", ShopStatus.UNKNOWN.value),
                "today_hours": BOOT_DATA["shop_state"].get("today_readable_hours", "Indisponível")
            }
        
        # Validação padrão através da função existente
        res = await validate_pickup(pickup_time, hours, current_datetime)
        
        # Informações adicionais sobre o estado da loja
        shop_state = BOOT_DATA["shop_state"] or {}
        res["shop_status"] = shop_state.get("status", ShopStatus.UNKNOWN.value)
        
        # Obter horário atual
        now = _dt.now(TZ)
        today_slots = get_todays_slots(now)
        readable_hours = ", ".join(
            f"{minutes_to_hms(s.start_minutes)}-{minutes_to_hms(s.end_minutes)}"
            for s in today_slots
        ) or "Encerrado hoje"
        
        res["today_hours"] = readable_hours
        
        # Mensagem completa sobre estado da loja
        res["shop_message"] = shop_state.get("message", "")
        
        # Se a loja estiver fechada, adiciona informações úteis
        if shop_state.get("status") == ShopStatus.CLOSED.value:
            res["next_open_time"] = shop_state.get("next_open_time")
            
            # Mais informativo sobre reabertura
            if shop_state.get("next_open_time"):
                res["message"] = (
                    f"Estamos fechados agora. Reabrimos às {shop_state.get('next_open_time')}. "
                    f"Horário hoje: {readable_hours}"
                )
            else:
                res["message"] = f"Estamos fechados hoje. {shop_state.get('message', '')}"
        
        # Se o horário for inválido, sugerir horários alternativos
        if not res.get("valid", False):
            avail_times = []
            for seg in hours.split("|"):
                tp = _extract_time_part(seg)
                if tp and "Disponível:Sim" in seg:
                    avail_times.append(tp)
            
            if avail_times:
                res["available_times"] = avail_times
                if not res.get("reason", "").endswith(f"Hoje: {readable_hours}"):
                    res["reason"] += f" Horário hoje: {readable_hours}"
                if avail_times:
                    res["reason"] += f" Horários disponíveis: {', '.join(avail_times)}"
                
        # Adiciona dicas extras para o agente
        if not res.get("valid", False):
            res["suggestions_for_agent"] = [
                "Confirme se o cliente quer escolher outro horário disponível",
                "Mencione claramente o horário de funcionamento atual",
                "Se estamos fechados, informe quando reabrimos"
            ]
            
        return res
    except Exception as e:
        # Recuperação graciosa de qualquer erro
        log.error(f"Erro na validação de pickup: {str(e)}")
        return {
            "valid": False,
            "reason": "Erro ao validar horário. Por favor, verifique o formato (HH:MM).",
            "error": str(e),
            "shop_status": BOOT_DATA["shop_state"].get("status", ShopStatus.UNKNOWN.value) if BOOT_DATA.get("shop_state") else ShopStatus.UNKNOWN.value
        }

# ─────────────────────────  Interpret Time  ─────────────────────
HM_REGEX  = re.compile(r"^(\d{1,2})[:|.](\d{2})$")
H_REGEX   = re.compile(r"^(\d{1,2})\s*h$")
HMM_REGEX = re.compile(r"^(\d{1,2})h(\d{2})$")
TIME_REGEX= re.compile(r"^\d{1,2}:\d{2}$")

@function_tool
async def interpret_time(time_expression: str, current_hour: int) -> dict:
    tx = time_expression.lower().strip()
    if TIME_REGEX.match(tx):
        h, m = map(int, tx.split(":"))
        return {"original": time_expression, "interpreted": f"{h:02}:{m:02}",
                "is_relative": False, "confidence": "high"}

    if tx in {"meio-dia","meio dia"}:
        return {"original": tx, "interpreted":"12:00", "is_relative":False,"confidence":"high"}
    if tx in {"meia-noite","meia noite"}:
        return {"original": tx, "interpreted":"00:00", "is_relative":False,"confidence":"high"}

    match = RELATIVE_TIME_REGEX.search(tx)
    if match:
        amt = int(match.group(1))
        unit = match.group(2)
        mins = amt if unit.startswith("min") else amt*60
        now = _dt.now(TZ)
        tgt = now + timedelta(minutes=mins)
        return {"original": tx, "interpreted": tgt.strftime("%H:%M"),
                "is_relative": True, "confidence": "high"}

    # fallback
    return {"original": tx, "interpreted": None, "confidence":"low",
            "message":"Não consegui interpretar. Pede confirmação."}

# ─────────────────────────  Order + Transfer  ─────────────────────
# Limite para evitar mega-pedidos
MAX_ITEMS = 8
MAX_CHAR_PER_ITEM = 100

@function_tool
async def order_confirmed(name: str, pickup_time: str,
                          items: list[str],
                          customizations: dict|None=None):
    mark("confirmed")  # Registra momento da confirmação
    
    # Guard-rails: limita número de itens
    if len(items) > MAX_ITEMS:
        items = items[:MAX_ITEMS]
        log.info(f"Pedido truncado para {MAX_ITEMS} itens")
    
    log.info("PEDIDO CONFIRMADO — %s %s — %s", name, pickup_time, items)
    
    # Processamento otimizado de personalizações
    formatted_items = []
    if customizations:
        for item, options in customizations.items():
            if isinstance(options, dict):
                opts_str = ", ".join(f"{k}: {v}" for k, v in options.items())
                formatted_items.append(f"{item} ({opts_str}"[:MAX_CHAR_PER_ITEM] + ")")
            else:
                formatted_items.append(f"{item} ({options}"[:MAX_CHAR_PER_ITEM] + ")")
    
    # Adiciona itens padrão (não personalizados)
    for item in items:
        if not any(item in custom_item for custom_item in formatted_items):
            formatted_items.append(item[:MAX_CHAR_PER_ITEM])
    
    # Simplifica formato do pedido
    transcription = f"{name} {pickup_time}\n" + "\n".join(formatted_items)
    
    # Extração otimizada do número de telefone
    phone_number = "unknown"
    job_ctx = TRANSFER_HUMAN_CONTEXT
    
    if job_ctx and job_ctx.room and job_ctx.room.name:
        parts = job_ctx.room.name.split("_")
        if len(parts) >= 2 and parts[1].startswith("+"):
            phone_number = parts[1]
            log.info(f"Telefone extraído: {phone_number}")
    
    # Envio do webhook com timeout reduzido
    payload = {"transcription": transcription, "phone": phone_number}
    
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                "https://hook.eu2.make.com/67puqsnvot28na9cget6444fiejy3go6?type=order-confirmed",
                json=payload, timeout=5) as r:
                response_data = await r.json(content_type=None)
                log.info(f"Resposta webhook: {response_data}")
        
        # Registra tempo total de confirmação
        if "start" in CALL_METRICS:
            ttco = (CALL_METRICS["confirmed"] - CALL_METRICS["start"]).total_seconds()
            log.info("TTCO %.2f s", ttco)
        
        return {
            "ok": True,
            "message": "Pedido confirmado",
            "transcription": transcription
        }
    except Exception as e:
        log.error(f"Erro no webhook: {str(e)}")
        return {
            "ok": True,  # Mantém ok=True para não confundir o cliente
            "message": "Pedido registado",
            "error_details": str(e)
        }

@function_tool
async def transfer_human(reason: str|None=None):
    log.info(f"TRANSFERÊNCIA PARA HUMANO → Motivo: {reason}")
    
    job_ctx = TRANSFER_HUMAN_CONTEXT
    if not job_ctx or not job_ctx.room:
        log.error("Transferência falhou: Contexto ou sala não disponível")
        return {"ok": False, "error": "Contexto ou sala não disponível"}
    
    # Get room name
    room_name = job_ctx.room.name
    log.info(f"Room name: {room_name}")
    
    # Extract phone number from room name
    phone_number = None
    if room_name and "_" in room_name:
        parts = room_name.split("_")
        if len(parts) >= 2:
            possible_phone = parts[1]
            if possible_phone.startswith("+"):
                phone_number = possible_phone
                log.info(f"Extracted phone number: {phone_number}")
    
    if not phone_number:
        # Fallback to direct participant search
        participant = next((p for p in job_ctx.room.participants
                          if p.identity.startswith("sip_")), None)
        if not participant:
            log.error("Transferência falhou: Não foi possível encontrar participante SIP")
            return {"ok": False, "error": "SIP participant não encontrado"}
        participant_identity = participant.identity
    else:
        # Construct SIP participant identity from phone number
        participant_identity = f"sip_{phone_number}"
    
    log.info(f"SIP participant identity: {participant_identity}")
    
    # Format phone number for transfer
    transfer_to = f"tel:{TRANSFER_PHONE_NUMBER}" if not TRANSFER_PHONE_NUMBER.startswith("tel:") else TRANSFER_PHONE_NUMBER
    
    # Execute the transfer
    req = TransferSIPParticipantRequest(
        participant_identity=participant_identity,
        room_name=room_name,
        transfer_to=transfer_to,
        play_dialtone=False)
        
    try:
        async with api.LiveKitAPI() as lk:
            await lk.sip.transfer_sip_participant(req)
        
        return {
            "ok": True,
            "message": "Transferência iniciada com sucesso",
            "participant": participant_identity,
            "room": room_name,
            "transfer_to": transfer_to
        }
    except Exception as e:
        log.error(f"Erro ao transferir para humano: {str(e)}")
        return {"ok": False, "error": str(e)}

# ─────────────────────────  Prompt  ─────────────────────────────
BASE_PROMPT = (
    "Função: és a operadora telefónica da Churrascaria Quitanda. "
    "Fala EXCLUSIVAMENTE em Português de Portugal, com sotaque europeu autêntico. "
    "CARACTERÍSTICAS LINGUÍSTICAS OBRIGATÓRIAS:\n"
    "- Vogais fechadas: 'telefone' (não 'telefôni')\n"
    "- 'e' final como 'ə' (schwa), nunca como 'i'\n"
    "- 's' final como 'sh' (chiante): 'boas' → 'boash'\n"
    "- 'r' pronunciado na garganta (uvular): 'carro', 'restaurante'\n"
    "- Usa 'tu' em vez de 'você': 'queres' (não 'você quer')\n"
    "- Usa 'pequeno-almoço', 'casa de banho', 'autocarro', 'telemóvel'\n"
    "- EVITA expressões brasileiras como: 'a gente', 'legal', 'tudo bem?', 'oi'\n"
    "- Usa 'olá', 'bom dia', 'estás', 'percebes', 'percebi', 'bom'\n"
    "- Usa verbos no presente contínuo com 'estar a': 'estou a preparar' (não 'estou preparando')\n"
    "- Usa 'se calhar' em vez de 'talvez', 'miúdo' em vez de 'garoto/menino'\n"
    "- Usa formas gramaticais típicas de Portugal: 'tens' (não 'você tem'), 'estás' (não 'está')\n\n"
    "VOCABULÁRIO PORTUGUÊS DE PORTUGAL (OBRIGATÓRIO):\n"
    "✅ telemóvel (NÃO use 'celular')\n"
    "✅ casa de banho (NÃO use 'banheiro')\n"
    "✅ autocarro (NÃO use 'ônibus')\n"
    "✅ comboio (NÃO use 'trem')\n"
    "✅ pequeno-almoço (NÃO use 'café da manhã')\n"
    "✅ frigorífico (NÃO use 'geladeira')\n"
    "✅ portátil (NÃO use 'laptop')\n"
    "✅ tasca, restaurante (NÃO use 'lanchonete')\n"
    "✅ casa de partilha (NÃO use 'república')\n"
    "✅ bicha, fila (NÃO use 'fila' com pronúncia brasileira)\n"
    "✅ peão (NÃO use 'pedestre')\n"
    "✅ empregado, funcionário (NÃO use 'atendente')\n"
    "✅ comando (NÃO use 'controle remoto')\n"
    "✅ talho (NÃO use 'açougue')\n"
    "✅ fala, diz (NÃO use 'fale', 'diga')\n\n"
    "Deixa o cliente liderar a conversa e expressar-se completamente. "
    "Não interrompas nem faças muitas perguntas de uma vez. "
    "Escuta atentamente e responde apenas ao que for perguntado. "
    "Pergunta apenas dados essenciais em falta no final (nome, hora de recolha, itens). "
    "Sê breve, objetivo e eficiente na confirmação do pedido."
)

def build_system_prompt() -> str:
    """Gera o prompt do sistema com dados do BOOT_DATA + nudge PT-PT."""
    now  = _dt.now(TZ)
    dia  = now.strftime("%A")
    hora = now.strftime("%H:%M")

    shop = BOOT_DATA["shop_state"] or {}
    menu = BOOT_DATA["menu"]       or "Menu indisponível"

    estado_txt = (
        "ABERTO até " + shop.get("next_close")
        if shop.get("status") == ShopStatus.OPEN.value
        else shop.get("message", "FECHADO")
    )

    # Inclui o bloco de nudging logo após o BASE_PROMPT
    base_prompt_with_nudge = (
        f"{BASE_PROMPT}\n\n{PT_PT_NUDGE_BLOCK}\n\n{PT_PT_EXAMPLES}"
    )

    # Adiciona regras específicas sobre pronúncia
    pt_pt_rules = """
    REGRAS OBRIGATÓRIAS DE SOTAQUE PORTUGUÊS EUROPEU:
    1. Pronúncia fechada das vogais átonas, especialmente o 'e' final (schwa)
    2. 's' final sempre como 'sh' chiante (ex: "boas" → "boash")
    3. 'r' gutural, na garganta, especialmente em início de palavra e 'rr'
    4. Entoação descendente típica do português europeu
    5. Ritmo mais pausado comparado ao português brasileiro
    6. Vogais nasais fechadas ('pão', 'bem')
    7. NUNCA usar entoação ascendente brasileira
    8. NUNCA usar 'r' tipo americano como no Brasil
    9. NUNCA prolongar vogais átonas
    10. NUNCA usar expressões ou pronúncia brasileiras
    """

    return (
        f"HORA ATUAL: {hora} ({dia})\n"
        f"ESTADO: {estado_txt}\n"
        f"HOJE: {shop.get('today_readable_hours','')}\n\n"
        f"{base_prompt_with_nudge}\n\n"
        f"{pt_pt_rules}\n\n"
        f"MENU:\n{menu}\n\n"
        "⚠️ Estes dados já estão carregados – evita repetir get_menu/get_shop_state.\n\n"
        "DIRETRIZES ADICIONAIS:\n"
        "1. Deixa SEMPRE o cliente liderar a conversa, sem interromper\n"
        "2. Responde apenas ao que for perguntado, sem adicionar informações extras\n"
        "3. Espera o cliente terminar de falar antes de perguntar algo\n"
        "4. Apenas no FINAL, se faltarem dados essenciais (nome, hora, itens), pergunta de forma breve\n"
        "5. Não faças perguntas em série - uma de cada vez\n"
        "6. Se o cliente mencionar um produto, assume que quer encomendar sem confirmar repetidamente\n"
        "7. Confirma o pedido apenas quando tiveres todos os dados, sem passos intermédios\n\n"
        "HORÁRIOS E DISPONIBILIDADE:\n"
        f"1. Respeita RIGOROSAMENTE o horário de funcionamento: {shop.get('today_readable_hours','')}\n"
        "2. Usa validate_pickup_combined para verificar se a hora pedida é válida\n"
        "3. Se o restaurante estiver fechado, informa claramente e sugere o próximo horário disponível\n"
        "4. Não aceites pedidos fora do horário de funcionamento\n"
        "5. Verifica sempre se o horário pedido está dentro do horário de funcionamento\n"
        "6. Sê consistente nas informações sobre horários e disponibilidade"
    )

# ─────────────────────────  Entrypoint  ─────────────────────────
TRANSFER_HUMAN_CONTEXT: JobContext|None = None   # para função acima

async def entrypoint(ctx: JobContext):
    global TRANSFER_HUMAN_CONTEXT
    TRANSFER_HUMAN_CONTEXT = ctx
    
    # Registra início da chamada para métricas
    mark("start")

    await boot_fetch_menu()
    
    # Registra o estado da loja no início
    now = _dt.now(TZ)
    shop_state = BOOT_DATA["shop_state"] or {}
    shop_status = shop_state.get("status", ShopStatus.UNKNOWN.value)
    log.info("Estado da loja no início: %s", shop_status)
    
    if shop_status == ShopStatus.OPEN.value:
        log.info("Loja aberta. Horário de encerramento: %s", shop_state.get("next_close", "desconhecido"))
    else:
        log.info("Loja fechada. Próxima abertura: %s", shop_state.get("next_open_time", "desconhecido"))
    
    # Configuração especializada para sotaque português europeu
    llm = openai.realtime.RealtimeModel(
        model="gpt-4o-realtime-preview", 
        voice="coral",  # Voz feminina que funciona bem com português europeu
        temperature=0.6,  # Mínimo aceitável pela API
        turn_detection=TurnDetection(
            type="semantic_vad", 
            eagerness="auto",
            create_response=True, 
            interrupt_response=True)
    )

    # Preparação para garantir sotaque português europeu
    pre_greeting = f"""
    {PT_PT_NUDGE_BLOCK}
    
    INSTRUÇÃO CRUCIAL: Esta conversa DEVE ocorrer em português europeu (de Portugal),
    NUNCA em português brasileiro. A pronúncia, vocabulário, fraseado e entoação
    devem ser EXCLUSIVAMENTE de Portugal.
    
    Exemplos comparativos:
    ❌ Brasileiro: "Olá, tudo bem? Você gostaria de fazer um pedido?"
    ✅ Português: "Olá, boa tarde. Queres fazer um pedido?"
    
    ❌ Brasileiro: "A gente está com uma promoção legal hoje"
    ✅ Português: "Temos uma promoção especial hoje"
    
    ❌ Brasileiro: "Qual é o seu telefone? Você quer levar para viagem?"
    ✅ Português: "Qual é o teu telemóvel? Queres levar para casa?"
    
    Responda APENAS com português europeu autêntico.
    """
    
    # Define as ferramentas disponíveis para o agente
    tools = [get_menu, get_shop_state,
             validate_pickup, validate_pickup_combined,
             interpret_time, order_confirmed, transfer_human]

    # Inicializa o agente com o prompt do sistema
    agent = Agent(instructions=build_system_prompt(), tools=tools)
    session = AgentSession(llm=llm)

    # Estabelece a conexão e inicia a sessão
    await ctx.connect()
    await session.start(agent, room=ctx.room)

    # Mensagem inicial com instruções detalhadas para sotaque português europeu
    initial_greeting = f"""
    {PT_PT_NUDGE_BLOCK}
    {pre_greeting}
    
    INSTRUÇÕES PRECISAS PARA PRONÚNCIA PORTUGUESA AUTÊNTICA:
    
    EXEMPLOS DE SAUDAÇÕES EM PORTUGUÊS EUROPEU:
    "Bom dia, em que posso ajudar?" (pronuncia: bum día, ẽ ki póssu ajudár)
    "Com certeza" (cum certêza, com 'e' fechado)
    "Está bem" (shtá bãi, com vogais nasais)
    "Faz favor" (fásh fvôr, 's' chiante e vogais reduzidas)
    
    Diz exatamente, com sotaque autêntico de Lisboa: 
    'Churrascaria Quitanda, bom dia. Em que posso ajudar-te?' 
    
    DETALHES FONÉTICOS OBRIGATÓRIOS:
    - 'Ch' como em 'chave', nunca como 'tch' brasileiro
    - 'rr' gutural na garganta (como em francês), nunca como no Brasil
    - 'posso ajudar-te' com vogais fechadas e 'r' na garganta
    - Ritmo pausado e melodia descendente típica do português europeu
    - 's' final SEMPRE como 'sh' chiante (nunca como 'ss' ou 'z')
    - Vogais átonas MUITO reduzidas (característica distintiva do português europeu)
    - 'o' final como 'u' muito curto
    - Entoação descendente no final da frase
    
    *** FALE COMO UMA PESSOA DE LISBOA, PORTUGAL - NUNCA DO BRASIL ***
    """
    
    await session.generate_reply(instructions=initial_greeting)

# ─────────────────────────  Runner  ────────────────────────────
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint,
                              worker_type=WorkerType.ROOM))
