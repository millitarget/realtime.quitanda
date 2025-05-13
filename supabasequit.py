#!/usr/bin/env python3
"""
supabasequit.py — LiveKit voice agent (PT-PT) with Supabase integration
Churrascaria Quitanda · Python 3.12 · Maio 2025
"""

from __future__ import annotations
import asyncio, logging, os, json, re, time as pytime
from dataclasses import dataclass
from datetime import datetime as _dt, timedelta, timezone, date, time
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
from supabase import create_client, Client

# ───────────────────────── Config base ─────────────────────────
load_dotenv(".env.local")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger("quitanda")
PT_PT_NUDGE_WORDS = (
    "telemóvel autocarro casa de banho pequeno-almoço elevador frigorífico "
    "comboio talho esquadra peão sumo portagem fato miúdos "
    "bicha apanhar formação estágio carta de condução "
    "carreira morada camisola autarquia boleia IVA"
)
# Bloco fantasma: não tem significado para o cliente,
# mas ajuda o TTS/OpenAI a detectar o dialecto correcto.
PT_PT_NUDGE_BLOCK = f"<nudge>{PT_PT_NUDGE_WORDS}</nudge>"
try:
    TZ = ZoneInfo("Europe/Lisbon")
except ZoneInfoNotFoundError:
    TZ = timezone.utc
log.info("Timezone forçado: %s", TZ)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# chave global com tudo o que vem do Supabase
BOOT_DATA: dict[str, Any] = {"menu": None, "hours": None, "shop_state": None, "slot_map": None}

# URL para buscar dados do menu/slots (padrão para testes quando variável ambiente não definida)
MAKE_URL = os.getenv("MENU_WEBHOOK_URL", "https://hook.eu2.make.com/test_endpoint_for_quitanda")
TRANSFER_PHONE_NUMBER = os.getenv("TRANSFER_PHONE_NUMBER", "+351933792547")

# ───────────────────────── Horário oficial ─────────────────────
# Mantido como fallback caso Supabase não esteja disponível
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
class PickupSlot:
    date: date         # 2025-05-08
    time: time         # 09:30
    available: bool    # True = Disponível:Sim
    
    def __str__(self) -> str:
        return f"{self.time.strftime('%H:%M')}" + (" [disponível]" if self.available else " [ocupado]")

SLOT_REGEX = re.compile(
    r"(?:Data:(?P<date>\d{4}-\d{2}-\d{2}))?.*?"
    r"Horário:(?P<time>\d{2}:\d{2}).*?"
    r"Disponível:(?P<disp>Sim|Não)", re.I
)

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

# ───────────────────────── Supabase functions ─────────────────────────
async def fetch_restaurant_hours() -> dict[str, list[TimeSlot]]:
    """
    Fetch restaurant hours from Supabase
    
    Returns:
        Dictionary mapping days to TimeSlot objects
    """
    if not supabase:
        log.warning("Supabase não está configurado. Usando horários padrão.")
        return SCHEDULE
    
    try:
        response = supabase.table("restaurant_hours").select("*").execute()
        hours_data = response.data
        
        if not hours_data:
            log.warning("Não foram encontrados horários na base de dados. Usando horários padrão.")
            return SCHEDULE
        
        # Convert to the same format as SCHEDULE
        result = {
            "monday": [], "tuesday": [], "wednesday": [], 
            "thursday": [], "friday": [], "saturday": [], "sunday": []
        }
        
        for record in hours_data:
            day = record.get("day_of_week", "").lower()
            if day in result:
                start_time = record.get("start_time")
                end_time = record.get("end_time")
                if start_time and end_time:
                    result[day].append(TimeSlot(
                        hms_to_minutes(start_time),
                        hms_to_minutes(end_time)
                    ))
        
        log.info(f"Horários carregados do Supabase: {sum(len(slots) for slots in result.values())} períodos")
        return result
    except Exception as e:
        log.error(f"Erro ao buscar horários do Supabase: {e}")
        return SCHEDULE

async def fetch_menu_items() -> list[dict]:
    """
    Fetch menu items from Supabase
    
    Returns:
        List of menu items with their options
    """
    if not supabase:
        log.warning("Supabase não está configurado. Usando menu padrão.")
        return []
    
    try:
        # Get all available menu items
        items_response = supabase.table("menu_items").select("*").eq("available", True).execute()
        menu_items = items_response.data
        
        if not menu_items:
            log.warning("Não foram encontrados itens de menu na base de dados.")
            return []
        
        # For each item, get its options
        result = []
        for item in menu_items:
            item_id = item.get("id")
            
            # Get options for this item
            options_response = supabase.table("menu_options").select("*").eq("item_id", item_id).execute()
            options_data = options_response.data
            
            # Process options
            options = {}
            for opt in options_data:
                opt_type = opt.get("option_type")
                opt_values = opt.get("option_values", [])
                
                if opt_type == "opcoes":
                    if isinstance(options, dict):
                        options["opcoes"] = opt_values
                    else:
                        options = {"opcoes": opt_values}
                elif opt_type == "picante":
                    if isinstance(options, dict):
                        options["picante"] = opt_values
                    else:
                        options = {"picante": opt_values}
            
            # Add options to item
            item["options"] = options
            result.append(item)
        
        log.info(f"Menu carregado do Supabase: {len(result)} itens")
        return result
    except Exception as e:
        log.error(f"Erro ao buscar menu do Supabase: {e}")
        return []

def format_menu_items(menu_items: list[dict]) -> str:
    """
    Format menu items as a string
    
    Args:
        menu_items: List of menu items from Supabase
        
    Returns:
        Formatted menu string
    """
    if not menu_items:
        return "Menu indisponível"
    
    sections = {}
    
    # Group items by category if available
    for item in menu_items:
        category = item.get("category", "OUTROS")
        if category not in sections:
            sections[category] = []
        
        price_str = f"{item.get('price', 0):.2f}€" if "price" in item else ""
        name = item.get("name", "Item sem nome")
        description = item.get("description", "")
        
        item_text = f"{name}" + (f" - {price_str}" if price_str else "")
        if description:
            item_text += f" ({description})"
            
        sections[category].append(item_text)
    
    # Format the complete menu
    result = []
    for category, items in sections.items():
        result.append(f"{category}:")
        result.extend([f"  {item}" for item in items])
        result.append("")  # Empty line between categories
    
    return "\n".join(result)

def format_menu_options(menu_items: list[dict]) -> dict:
    """
    Format menu options as expected by the application
    
    Args:
        menu_items: List of menu items from Supabase
        
    Returns:
        Dictionary of menu options
    """
    result = {}
    
    for item in menu_items:
        name = item.get("name", "").lower()
        options = item.get("options", {})
        
        if options and name:
            result[name] = options
    
    return result 

# ───────────────────────── BOOT fetch (1 vez) ──────────────────
async def boot_fetch_menu() -> None:
    try:
        # If Supabase is configured, use it
        if supabase:
            log.info("Buscando dados do Supabase...")
            
            # Fetch restaurant hours
            schedule = await fetch_restaurant_hours()
            
            # Fetch menu items
            menu_items = await fetch_menu_items()
            
            # Format menu as a string for display
            menu_str = format_menu_items(menu_items)
            
            # Format menu options for the application
            menu_options = format_menu_options(menu_items)
            
            # Generate hours string (similar to make.com format for compatibility)
            hours_str = generate_hours_string(schedule)
            
            # Store in BOOT_DATA
            BOOT_DATA["menu"] = menu_str
            BOOT_DATA["hours"] = hours_str
            BOOT_DATA["menu_options"] = menu_options
            
            # Fetch time slots directly from the time_slots table
            BOOT_DATA["slot_map"] = await fetch_time_slots()
            
            # Use the schedule for slot calculations if no slots found
            if not BOOT_DATA["slot_map"]:
                today = _dt.now(TZ).date()
                BOOT_DATA["slot_map"] = generate_slot_map(schedule, today)
                
            BOOT_DATA["shop_state"] = await compute_shop_state(schedule)
            
            log.info("BOOT_DATA carregado do Supabase — menu %d chars, hours %d chars, slots %d dias",
                    len(str(BOOT_DATA['menu'])), len(str(BOOT_DATA['hours'])), 
                    len(BOOT_DATA["slot_map"]))
            return
            
        # Fallback to make.com if Supabase not configured
        log.info("Supabase não configurado, usando Make.com como fallback")
        async with aiohttp.ClientSession() as sess:
            async with sess.get(MAKE_URL, timeout=8) as resp:
                raw = await resp.json(content_type=None)

        src = raw.get("dynamic_variables", raw)
        BOOT_DATA["menu"]  = src.get("menu_items", "")
        BOOT_DATA["hours"] = src.get("hoursanddate", "")
        today = _dt.now(TZ).date()
        BOOT_DATA["slot_map"] = parse_hoursanddate(BOOT_DATA["hours"], today)
        BOOT_DATA["shop_state"] = await compute_shop_state()
        log.info("BOOT_DATA carregado do Make — menu %d chars, hours %d chars, slots %d dias",
                len(str(BOOT_DATA['menu'])), len(str(BOOT_DATA['hours'])), 
                len(BOOT_DATA["slot_map"]))
    except Exception as e:
        log.error(f"Erro ao carregar dados: {e}")
        
        # Gerar dados de fallback para teste
        if not BOOT_DATA["menu"]:
            BOOT_DATA["menu"] = """
            ENTRADAS:
            Pão e Manteiga - 2,50€
            Couvert completo - 4,50€
            Caldo Verde - 3,50€
            
            PRATOS PRINCIPAIS:
            Picanha - 18,90€
            Costela de Boi - 17,50€
            Frango Grelhado - 12,90€
            Bacalhau à Brás - 16,50€
            
            BEBIDAS:
            Água - 2,00€
            Refrigerante - 2,50€
            Cerveja - 3,00€
            Vinho da Casa - 9,90€
            """
        
        # Criar dados de horário fictícios se não carregados
        if not BOOT_DATA["hours"]:
            today = _dt.now(TZ).date()
            tomorrow = today + timedelta(days=1)
            
            # Gerar horários fictícios para slots
            slots_data = []
            for day, date_val in [(today, today.isoformat()), (tomorrow, tomorrow.isoformat())]:
                for h in [12, 13, 19, 20]:
                    for m in [0, 30]:
                        time_str = f"{h:02}:{m:02}"
                        # Verificar se o horário já passou para hoje
                        now = _dt.now(TZ)
                        slot_dt = _dt.combine(day, _dt.strptime(time_str, "%H:%M").time())
                        
                        available = "Sim" if slot_dt > now else "Não"
                        slots_data.append(f"Data:{date_val} Horário:{time_str} Disponível:{available}")
            
            # Concatenar todos os slots em uma string
            BOOT_DATA["hours"] = "\n".join(slots_data)
            
            # Parseando os horários
            BOOT_DATA["slot_map"] = parse_hoursanddate(BOOT_DATA["hours"], today)
        
        # Criar estado da loja se não definido
        if not BOOT_DATA["shop_state"]:
            BOOT_DATA["shop_state"] = await compute_shop_state()
        
        log.info("BOOT_DATA carregado com fallback — slots %d dias",
                len(BOOT_DATA["slot_map"]))

def generate_hours_string(schedule: dict[str, list[TimeSlot]]) -> str:
    """
    Generate hours string in the expected format for compatibility with existing code
    
    Args:
        schedule: Dictionary of day -> TimeSlot list
        
    Returns:
        Formatted hours string
    """
    today = _dt.now(TZ).date()
    tomorrow = today + timedelta(days=1)
    
    # Generate slots for today and tomorrow
    result = []
    
    for day_offset in range(7):  # Generate for a week
        target_date = today + timedelta(days=day_offset)
        weekday = target_date.strftime("%A").lower()
        
        # Get slots for this day
        day_slots = schedule.get(weekday, [])
        if not day_slots:
            continue
            
        # Generate slots at 30 minute intervals within operating hours
        for slot in day_slots:
            start_mins = slot.start_minutes
            end_mins = slot.end_minutes
            
            for mins in range(start_mins, end_mins, 30):
                time_str = minutes_to_hms(mins)
                
                # Check if this slot is in the past (for today)
                is_past = False
                if target_date == today:
                    now = _dt.now(TZ)
                    now_mins = now.hour * 60 + now.minute
                    if mins <= now_mins:
                        is_past = True
                
                available = "Não" if is_past else "Sim"
                result.append(f"Data:{target_date.isoformat()} Horário:{time_str} Disponível:{available}")
    
    return "\n".join(result)

def generate_slot_map(schedule: dict[str, list[TimeSlot]], start_date: date) -> dict[date, list[PickupSlot]]:
    """
    Generate slot map based on operating hours
    
    Args:
        schedule: Dictionary of day -> TimeSlot list
        start_date: Starting date to generate slots from
        
    Returns:
        Dictionary mapping dates to available pickup slots
    """
    slot_map = {}
    now = _dt.now(TZ)
    now_mins = now.hour * 60 + now.minute
    
    # Generate slots for the next 7 days
    for day_offset in range(7):
        target_date = start_date + timedelta(days=day_offset)
        weekday = target_date.strftime("%A").lower()
        
        # Get operating hours for this day
        day_slots = schedule.get(weekday, [])
        if not day_slots:
            continue
        
        # Generate slots at 30 minute intervals
        day_pickup_slots = []
        for slot in day_slots:
            start_mins = slot.start_minutes
            end_mins = slot.end_minutes
            
            for mins in range(start_mins, end_mins, 30):
                slot_time = time(hour=mins // 60, minute=mins % 60)
                
                # Check if this slot is in the past (for today)
                is_past = False
                if target_date == start_date and mins <= now_mins:
                    is_past = True
                
                day_pickup_slots.append(PickupSlot(
                    date=target_date,
                    time=slot_time,
                    available=not is_past
                ))
        
        if day_pickup_slots:
            slot_map[target_date] = day_pickup_slots
    
    return slot_map 

# ───────────────────────── Helpers horário ─────────────────────
def get_todays_slots(now: _dt, custom_schedule: dict = None) -> list[TimeSlot]:
    """Get slots for today, optionally using a custom schedule"""
    schedule = custom_schedule or SCHEDULE
    return schedule.get(now.strftime("%A").lower(), [])

def get_slots_for_day(d: date, *, only_available: bool = True) -> list[PickupSlot]:
    """
    Retorna slots para um dia específico, filtrando por disponibilidade se necessário.
    
    NOTA PARA O ASSISTENTE:
    - Esta função busca os slots disponíveis para um dia específico
    - Slots são horários específicos para reservas (ex: 12:00, 12:30, 13:00...)
    - Cada slot tem um status de disponibilidade (disponível ou ocupado)
    - Quando interpreta respostas de disponibilidade, considere:
      * Verificar se há slots para o dia solicitado (pode estar fechado)
      * Se disponível para o horário exato solicitado
      * Se não, sugerir alternativas próximas de forma natural
      
    Args:
        d: Data para verificar disponibilidade
        only_available: Se True, retorna apenas slots disponíveis
        
    Returns:
        Lista de PickupSlot para o dia especificado
    """
    slots = BOOT_DATA.get("slot_map", {}).get(d, [])
    return [s for s in slots if s.available] if only_available else slots

def find_next_available_time(now: _dt, custom_schedule: dict = None) -> Optional[_dt]:
    """Find next available time using the provided or default schedule"""
    schedule = custom_schedule or SCHEDULE
    mins = now.hour*60 + now.minute
    
    # hoje
    for s in get_todays_slots(now, schedule):
        if s.start_minutes > mins:
            return now.replace(hour=s.start_minutes//60,
                               minute=s.start_minutes%60,
                               second=0, microsecond=0)
    # próximos dias
    for i in range(1, 8):
        nd = now + timedelta(days=i)
        weekday = nd.strftime("%A").lower()
        sl = schedule.get(weekday, [])
        if sl:
            f = sl[0]
            return nd.replace(hour=f.start_minutes//60,
                              minute=f.start_minutes%60,
                              second=0, microsecond=0)
    return None

def nearest_slot(dt_req: _dt, max_delta_min: int = 15) -> Optional[PickupSlot]:
    """
    Devolve o slot disponível mais próximo dentro de ±max_delta_min.
    
    NOTA PARA O ASSISTENTE:
    - Use esta função para encontrar alternativas próximas quando um horário não está disponível
    - Ao sugerir alternativas, use linguagem natural:
      * "Temos disponibilidade 15 minutos mais tarde, às 20:15"
      * "Temos uma mesa disponível um pouco antes, às 19:45"
    - Se não houver slots próximos, considere verificar outros dias
    
    Args:
        dt_req: Data e hora solicitada
        max_delta_min: Diferença máxima em minutos para considerar (padrão: 15 min)
        
    Returns:
        O slot disponível mais próximo, ou None se não houver dentro do limite
    """
    day_slots = get_slots_for_day(dt_req.date())
    if not day_slots:
        return None
        
    target = dt_req.time()
    best = min(
        day_slots,
        key=lambda s: abs((_dt.combine(date.min, s.time) -
                           _dt.combine(date.min, target)).total_seconds()),
        default=None,
    )
    
    if best and abs(
        (_dt.combine(date.min, best.time) -
         _dt.combine(date.min, target)).total_seconds()
    ) <= max_delta_min * 60:
        return best
    return None

async def compute_shop_state(custom_schedule: dict = None) -> dict:
    """Compute shop state using the provided or default schedule and time slots from Supabase."""
    schedule = custom_schedule or SCHEDULE
    now = _dt.now(TZ)
    now_date = now.date()
    now_time = now.time()
    now_mins = now.hour*60 + now.minute
    
    # Check if we have slots from Supabase for today
    today_slots_from_db = BOOT_DATA.get("slot_map", {}).get(now_date, [])
    
    # Get regular operating hours for today from schedule
    slots = get_todays_slots(now, schedule)
    readable = ", ".join(f"{minutes_to_hms(s.start_minutes)}-{minutes_to_hms(s.end_minutes)}"
                         for s in slots) or "Encerrado hoje"
    
    # Get available slots for today (for display purposes)
    available_slots_txt = ", ".join(s.time.strftime("%H:%M") for s in today_slots_from_db if s.available) if today_slots_from_db else "Nenhum"
    
    # Calculate the total slots and available slots counts
    total_slots = len(today_slots_from_db) if today_slots_from_db else 0
    available_slots = sum(1 for s in today_slots_from_db if s.available) if today_slots_from_db else 0
    
    # Format for display
    slots_status = f"{available_slots}/{total_slots} disponíveis" if total_slots > 0 else "Sem slots configurados"
    
    # Try to find a current active slot from Supabase
    current_db_slot = None
    if today_slots_from_db:
        for slot in today_slots_from_db:
            # Check if current time is within 30 minutes of this slot (slots are 30 min apart)
            slot_mins = slot.time.hour * 60 + slot.time.minute
            if abs(slot_mins - now_mins) < 30:
                current_db_slot = slot
                break
    
    # Check against operating hours
    is_open = False
    next_close_time = None
    for s in slots:
        if s.start_minutes <= now_mins < s.end_minutes:
            is_open = True
            next_close_time = minutes_to_hms(s.end_minutes)
            break
    
    # If we have a current slot from Supabase, use its availability
    if current_db_slot:
        # If the database has a slot for current time, its availability overrides schedule
        # (even if schedule says we're closed but there's an available slot)
        if current_db_slot.available:
            is_open = True
            
            # Try to find the next slot to determine closing time
            next_slot_idx = -1
            for i, slot in enumerate(today_slots_from_db):
                if slot.time > now_time:
                    next_slot_idx = i
                    break
            
            # If we didn't find a next slot, use the schedule's closing time
            if next_slot_idx == -1 and next_close_time:
                pass  # Keep the next_close_time from schedule
            elif next_slot_idx >= 0:
                # If the next slot exists but is unavailable, that's our closing time
                next_slot = today_slots_from_db[next_slot_idx]
                if not next_slot.available:
                    next_close_time = next_slot.time.strftime("%H:%M")
        else:
            # Current slot exists but is unavailable - we're closed
            is_open = False
            
            # Find the next available slot today
            next_available = next((s for s in today_slots_from_db 
                                 if s.time > now_time and s.available), None)
            if next_available:
                next_open_time = next_available.time.strftime("%H:%M")
                msg = f"Estamos fechados agora. Próxima abertura: {next_open_time}"
                return {
                    "status": ShopStatus.CLOSED.value,
                    "next_open_time": next_open_time,
                    "today_readable_hours": readable,
                    "message": msg,
                    "available_slots": available_slots_txt,
                    "slots_status": slots_status,
                    "status_message": f"FECHADO - Próxima abertura: {next_open_time}"
                }
    
    # If we determined we're open, return open status
    if is_open:
        return {
            "status": ShopStatus.OPEN.value,
            "next_close": next_close_time,
            "today_readable_hours": readable,
            "available_slots": available_slots_txt,
            "slots_status": slots_status,
            "status_message": f"ABERTO até {next_close_time}"
        }

    # If we're closed, find next opening time
    nxt = find_next_available_time(now, schedule)
    msg = "Estamos fechados."
    nxt_str = None
    if nxt:
        nxt_str = nxt.strftime("%H:%M")
        msg = f"Estamos fechados agora. Próxima abertura: {nxt_str}"
        
    # Check if we have any upcoming available slots from Supabase
    upcoming_db_slot = None
    if today_slots_from_db:
        upcoming_db_slot = next((s for s in today_slots_from_db 
                                if s.time > now_time and s.available), None)
        if upcoming_db_slot:
            nxt_str = upcoming_db_slot.time.strftime("%H:%M")
            msg = f"Estamos fechados agora. Próxima abertura: {nxt_str}"
    
    return {
        "status": ShopStatus.CLOSED.value,
        "next_open_time": nxt_str,
        "today_readable_hours": readable,
        "message": msg,
        "available_slots": available_slots_txt,
        "slots_status": slots_status,
        "status_message": f"FECHADO - {msg}" if nxt_str else "FECHADO"
    }

# ───────────────────────── Ferramentas LLM ─────────────────────
@function_tool
async def get_menu(hours_only: bool=False) -> dict:
    """Devolve menu/hours do BOOT_DATA (sem HTTP)."""
    data = {
        "menu_items": BOOT_DATA["menu"] or "Menu indisponível",
        "hoursanddate": BOOT_DATA["hours"] or "Horários indisponíveis",
        "menu_options": BOOT_DATA.get("menu_options", {}),
    }
    return {"hoursanddate": data["hoursanddate"]} if hours_only else data

@function_tool
async def get_shop_state(date_string: str="") -> dict:
    """
    Retorna o estado do restaurante (aberto/fechado) e informações sobre 
    disponibilidade de slots para reservas.
    
    Args:
        date_string: Data opcional no formato YYYY-MM-DD (default: data atual)
    
    Returns:
        Dicionário com estado do restaurante, horários e disponibilidade de slots
    """
    state = BOOT_DATA["shop_state"] or {"status": ShopStatus.UNKNOWN.value}
    
    if date_string:
        try:
            # Se uma data específica foi solicitada, verificar disponibilidade para essa data
            target_date = _dt.strptime(date_string, "%Y-%m-%d").date()
            
            # Verificar se há slots disponíveis para a data solicitada
            all_slots = BOOT_DATA.get("slot_map", {}).get(target_date, [])
            available_slots = [s for s in all_slots if s.available]
            
            # Formatar para exibição
            weekday = target_date.strftime("%A").lower()
            schedule_slots = SCHEDULE.get(weekday, [])
            
            if not schedule_slots:
                return {
                    "status": ShopStatus.CLOSED.value,
                    "date": date_string,
                    "message": f"Restaurante fechado neste dia: {date_string}",
                    "status_message": "FECHADO - Dia sem funcionamento",
                    "has_slots": False,
                    "slots_available": []
                }
            
            # Verificar se há slots disponíveis para a data
            if available_slots:
                slots_txt = [s.time.strftime("%H:%M") for s in available_slots]
                slots_status = f"{len(available_slots)}/{len(all_slots)} slots disponíveis"
                
                return {
                    "status": ShopStatus.OPEN.value,
                    "date": date_string,
                    "message": f"Restaurante aberto neste dia: {date_string}",
                    "status_message": f"ABERTO - {slots_status}",
                    "has_slots": True,
                    "slots_available": slots_txt,
                    "slots_status": slots_status
                }
            else:
                # Se não há slots disponíveis mas o restaurante funciona neste dia
                operating_hours = ", ".join(f"{minutes_to_hms(s.start_minutes)}-{minutes_to_hms(s.end_minutes)}" 
                                           for s in schedule_slots)
                
                return {
                    "status": ShopStatus.OPEN.value,
                    "date": date_string,
                    "message": f"Restaurante aberto neste dia, mas sem slots disponíveis",
                    "status_message": "ABERTO - Slots esgotados",
                    "operating_hours": operating_hours,
                    "has_slots": False,
                    "slots_available": []
                }
        except ValueError:
            # Se a data for inválida, retornar erro
            return {
                "status": ShopStatus.UNKNOWN.value,
                "error": "Formato de data inválido. Use YYYY-MM-DD",
                "status_message": "ERRO - Formato de data inválido"
            }
    
    # Se estamos apenas verificando o estado atual, retornar as informações de BOOT_DATA
    if "status_message" not in state:
        # Adicionar status_message se ainda não existir (para compatibilidade com versões antigas)
        if state.get("status") == ShopStatus.OPEN.value:
            state["status_message"] = f"ABERTO até {state.get('next_close', 'N/A')}"
        else:
            next_open = state.get("next_open_time", "")
            state["status_message"] = f"FECHADO - Próxima abertura: {next_open}" if next_open else "FECHADO"
    
    return state

@function_tool
async def refresh_data_from_supabase() -> dict:
    """
    Refresh data from Supabase database.
    Use this to get the latest menu and hours information.
    """
    try:
        # Reload everything from Supabase
        await boot_fetch_menu()
        
        # Explicitly refresh the time slots to ensure they're up-to-date
        BOOT_DATA["slot_map"] = await fetch_time_slots()
        
        return {
            "success": True,
            "message": "Dados atualizados com sucesso do Supabase",
            "timestamp": _dt.now(TZ).isoformat(),
            "slots_count": sum(len(slots) for slots in BOOT_DATA["slot_map"].values())
        }
    except Exception as e:
        log.error(f"Erro ao atualizar dados do Supabase: {e}")
        return {
            "success": False,
            "message": f"Erro ao atualizar dados: {str(e)}",
            "timestamp": _dt.now(TZ).isoformat()
        }

# ─────────────────────────  Validate Pick-up  ───────────────────
RELATIVE_TIME_REGEX = re.compile(r"daqui\s+a\s+(\d+)\s*(minutos|mins|min|horas|hrs|h)")

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
                          hoursanddate: str,
                          current_datetime: str) -> dict:
    """Wrapper para manter compatibilidade com a interface existente."""
    return await validate_pickup_new(pickup_time, hoursanddate, current_datetime)

async def validate_pickup_new(pickup_time: str, hoursanddate: str, current_datetime: str) -> dict:
    """Versão melhorada que usa o slot_map."""
    # ➊ primeiro, interpreta a hora
    interp = await interpret_time_new(pickup_time, current_datetime)
    if not interp["interpreted"]:
        return {"valid": False, "reason": interp["message"]}

    pickup_dt = _dt.fromisoformat(interp["dt"])
    today_slots = get_slots_for_day(pickup_dt.date())

    if not today_slots:
        return {"valid": False, "reason": "Estamos encerrados nesse dia.", "is_operating_hours_issue": True}

    # hora exata
    exact = next((s for s in today_slots if s.time.strftime("%H:%M") == interp["interpreted"]), None)
    if exact:
        return {"valid": True}

    # tenta slot próximo
    close = nearest_slot(pickup_dt)
    if close:
        return {
            "valid": False,
            "reason": f"Às {pickup_time} já não temos vaga.",
            "available_slots": [s.time.strftime("%H:%M") for s in today_slots],
            "suggested_slot": close.time.strftime("%H:%M"),
            "is_operating_hours_issue": False
        }

    # completamente fora do horário do dia
    day_readable = ", ".join(s.time.strftime("%H:%M") for s in today_slots)
    return {"valid": False,
            "reason": f"Nesse dia só temos: {day_readable}",
            "is_operating_hours_issue": True}

@function_tool
async def validate_pickup_combined(pickup_time: str,
                                   current_datetime: str,
                                   raw_expression: str=None) -> dict:
    """Wrapper para manter compatibilidade com a interface existente."""
    return await validate_pickup_combined_new(pickup_time, current_datetime, raw_expression)

async def validate_pickup_combined_new(pickup_time: str, current_datetime: str, raw_expression: str = None) -> dict:
    """Versão melhorada usando slot_map."""
    # ➊ primeiro, interpreta a hora
    interp = await interpret_time_new(pickup_time, current_datetime)
    if not interp["interpreted"]:
        return {"valid": False, "reason": interp["message"]}

    # Check shop status
    shop_status = BOOT_DATA["shop_state"]["status"]
    if shop_status == ShopStatus.CLOSED.value:
        return {
            "valid": False,
            "shop_status": ShopStatus.CLOSED.value,
            "next_open_time": BOOT_DATA["shop_state"].get("next_open_time"),
            "reason": "Estamos fechados agora. Não podemos aceitar pedidos.",
            "is_operating_hours_issue": True
        }
    
    pickup_dt = _dt.fromisoformat(interp["dt"])
    today_slots = get_slots_for_day(pickup_dt.date())
    operating_hours = BOOT_DATA["shop_state"].get('today_readable_hours','')

    if not today_slots:
        return {
            "valid": False, 
            "shop_status": shop_status,
            "reason": "Estamos encerrados nesse dia.",
            "is_operating_hours_issue": True
        }

    # hora exata
    exact = next((s for s in today_slots if s.time.strftime("%H:%M") == interp["interpreted"]), None)
    if exact:
        return {
            "valid": True,
            "shop_status": shop_status
        }

    # tenta slot próximo
    close = nearest_slot(pickup_dt)
    if close:
        return {
            "valid": False,
            "shop_status": shop_status,
            "reason": f"Às {pickup_time} já não temos disponibilidade, mas podemos oferecer às {close.time.strftime('%H:%M')}",
            "available_slots": [s.time.strftime("%H:%M") for s in today_slots],
            "suggested_slot": close.time.strftime("%H:%M"),
            "is_operating_hours_issue": False
        }

    # completamente fora do horário/sem slots
    day_readable = ", ".join(s.time.strftime("%H:%M") for s in today_slots)
    return {
        "valid": False,
        "shop_status": shop_status,
        "reason": f"Nesse dia só temos disponibilidade às: {day_readable}",
        "available_slots": [s.time.strftime("%H:%M") for s in today_slots],
        "is_operating_hours_issue": True,
        "operating_hours_explanation": f"Os horários de funcionamento são: {operating_hours}"
    } 

# ─────────────────────────  Interpret Time  ─────────────────────
HM_REGEX  = re.compile(r"^(\d{1,2})[:|.](\d{2})$")
H_REGEX   = re.compile(r"^(\d{1,2})\s*h$")
HMM_REGEX = re.compile(r"^(\d{1,2})h(\d{2})$")
TIME_REGEX= re.compile(r"^\d{1,2}:\d{2}$")

# Expressões coloquiais de tempo em português de Portugal
NUMBER_WORDS = {
    "meia": 30,  # "sete e meia"
    "um": 1, "uma": 1, "dois": 2, "duas": 2, "três": 3, "quatro": 4, "cinco": 5,
    "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10, "onze": 11, "doze": 12,
}
# "da manhã / tarde / noite" → deslocamento para 24 h
SHIFT_12H = {"manhã": 0, "tarde": 12, "noite": 18}

# Expressões coloquiais de tempo
COLLOQ_REGEXES = [
    # sete e um quarto → 07:15
    (re.compile(r"(\d{1,2})\s*e\s*um\s*quarto"), lambda h: (int(h), 15)),
    # sete e meia → 07:30
    (re.compile(r"(\d{1,2})\s*e\s*meia"), lambda h: (int(h), 30)),
    # dez para as oito → 07:50
    (re.compile(r"dez\s+para\s+as?\s+(\d{1,2})"), lambda h: ((int(h)-1)%24, 50)),
    # um quarto para as oito → 07:45
    (re.compile(r"um\s*quarto\s+para\s+as?\s+(\d{1,2})"), lambda h: ((int(h)-1)%24, 45)),
    # Acrescentando expressões por extenso - sete e meia
    (re.compile(r"(um|uma|dois|duas|três|quatro|cinco|seis|sete|oito|nove|dez|onze|doze)\s+e\s+meia"), 
     lambda h: (NUMBER_WORDS[h], 30)),
    # um quarto para as oito (por extenso)
    (re.compile(r"um\s*quarto\s+para\s+as?\s+(um|uma|dois|duas|três|quatro|cinco|seis|sete|oito|nove|dez|onze|doze)"), 
     lambda h: ((NUMBER_WORDS[h]-1)%24, 45)),
    # dez para as oito (por extenso)
    (re.compile(r"dez\s+para\s+as?\s+(um|uma|dois|duas|três|quatro|cinco|seis|sete|oito|nove|dez|onze|doze)"), 
     lambda h: ((NUMBER_WORDS[h]-1)%24, 50)),
    # sete e quinze/sete e quarenta e cinco
    (re.compile(r"(\d{1,2})\s*e\s*quinze"), lambda h: (int(h), 15)),
    (re.compile(r"(\d{1,2})\s*e\s*quarenta\s*e\s*cinco"), lambda h: (int(h), 45)),
    # Expressões por extenso + variações populares em português
    (re.compile(r"(um|uma|dois|duas|três|quatro|cinco|seis|sete|oito|nove|dez|onze|doze)\s+e\s+quinze"), 
     lambda h: (NUMBER_WORDS[h], 15)),
    (re.compile(r"(um|uma|dois|duas|três|quatro|cinco|seis|sete|oito|nove|dez|onze|doze)\s+e\s+quarenta\s*e\s*cinco"), 
     lambda h: (NUMBER_WORDS[h], 45)),
]

@function_tool
async def interpret_time(time_expression: str, current_hour: int) -> dict:
    """Função existente, mas implementada com a versão melhorada."""
    # Adapter para manter a assinatura atual mas usar a implementação melhorada
    # Converter current_hour para um datetime atual
    now = _dt.now(TZ)
    # Se o valor atual for diferente do horário do sistema, ajustar
    if now.hour != current_hour and 0 <= current_hour < 24:
        now = now.replace(hour=current_hour)
    
    ref_dt = now.isoformat()
    
    # Chama a implementação melhorada que lida com expressões coloquiais
    result = await interpret_time_new(time_expression, ref_dt)
    
    # Remove campos extras que não existiam na implementação original
    # para manter compatibilidade com código existente
    if "dt" in result:
        del result["dt"]
    if "day_offset" in result:
        del result["day_offset"]
    
    return result

async def interpret_time_new(time_expression: str, current_datetime: str = None) -> dict:
    """
    Versão melhorada que oferece suporte a expressões coloquiais portuguesas.
    Devolve:
        - interpreted: "HH:MM"
        - dt: datetime (Europe/Lisbon)
        - day_offset: 0 hoje, 1 amanhã…
        - context: informação contextual sobre o período do dia
        - confidence: grau de confiança na interpretação
    """
    if current_datetime:
        ref = _dt.fromisoformat(current_datetime).astimezone(TZ)
    else:
        ref = _dt.now(TZ)

    tx = time_expression.lower().strip()
    
    # ----------------
    # RECOMENDAÇÃO PARA O ASSISTENTE:
    # Quando receberes o resultado desta função, considera:
    # 1. A confiança na interpretação (alta/média/baixa)
    # 2. O contexto (se disponível) para falar de forma natural
    # 3. O day_offset para referir corretamente o dia (hoje/amanhã/etc.)
    # ----------------
    
    # ---------------- exact HH:MM ----------------
    m = re.match(r"^(\d{1,2})[:h.](\d{2})$", tx)
    if m:
        h, mnt = map(int, m.groups())
        dt = ref.replace(hour=h, minute=mnt, second=0, microsecond=0)
        if dt < ref:  # se já passou, assume "amanhã"
            dt += timedelta(days=1)
        return _ok(tx, dt, "high")

    # ---------------- meio-dia/meia-noite ----------------
    if tx in {"meio-dia","meio dia"}:
        dt = ref.replace(hour=12, minute=0, second=0, microsecond=0)
        if dt < ref:
            dt += timedelta(days=1)
        return _ok(tx, dt, "high", context="Hora de almoço")
    
    if tx in {"meia-noite","meia noite"}:
        dt = ref.replace(hour=0, minute=0, second=0, microsecond=0)
        dt += timedelta(days=1)  # meia-noite é sempre a próxima
        return _ok(tx, dt, "high", context="Fim da noite")

    # ---------------- "às oito da noite" ----------------
    m = re.match(r"(?:às|as|á|a)\s+(\d{1,2})(?:[:h](\d{2}))?\s*(da (manhã|tarde|noite))?", tx)
    if m:
        h = int(m.group(1))
        mnt = int(m.group(2) or 0)
        shift = SHIFT_12H.get(m.group(4) or "", 0)
        
        # Corrigir horas fora do intervalo 0-23
        if m.group(4) and h > 12:
            # Se já especificou "da noite" e a hora é > 12, não adiciona shift
            h = h % 12
        else:
            # Caso contrário, aplica o shift normalmente
            h = (h % 12) + shift if m.group(4) else h
            
        # Garantir que a hora está no intervalo 0-23
        h = min(23, max(0, h))
        
        dt = ref.replace(hour=h, minute=mnt, second=0, microsecond=0)
        if dt < ref:
            dt += timedelta(days=1)
        
        context = f"Período da {m.group(4)}" if m.group(4) else None
        return _ok(tx, dt, "medium", context=context)

    # ---------------- expressões coloquiais ----------------
    for rgx, fmt in COLLOQ_REGEXES:
        mm = rgx.search(tx)
        if mm:
            h, mnt = fmt(mm.group(1))
            dt = ref.replace(hour=h, minute=mnt, second=0, microsecond=0)
            if dt < ref:
                dt += timedelta(days=1)
            return _ok(tx, dt, "medium")
            
    # ---------------- Períodos do dia ----------------
    # Meal-based times
    if "almoço" in tx or "almoco" in tx:
        if "cedo" in tx or "início" in tx or "inicio" in tx:
            dt = ref.replace(hour=12, minute=0, second=0, microsecond=0)
        elif "tarde" in tx or "fim" in tx:
            dt = ref.replace(hour=14, minute=0, second=0, microsecond=0)
        else:
            dt = ref.replace(hour=13, minute=0, second=0, microsecond=0)
        
        if dt < ref:
            dt += timedelta(days=1)
        return _ok(tx, dt, "medium", context="Horário de almoço")
    
    if "jantar" in tx:
        if "cedo" in tx or "início" in tx or "inicio" in tx:
            dt = ref.replace(hour=19, minute=0, second=0, microsecond=0)
        elif "tarde" in tx or "fim" in tx:
            dt = ref.replace(hour=21, minute=0, second=0, microsecond=0)
        else:
            dt = ref.replace(hour=20, minute=0, second=0, microsecond=0)
            
        if dt < ref:
            dt += timedelta(days=1)
        return _ok(tx, dt, "medium", context="Horário de jantar")
    
    # Time of day references
    if "fim da tarde" in tx or "final da tarde" in tx:
        dt = ref.replace(hour=18, minute=0, second=0, microsecond=0)
        if dt < ref:
            dt += timedelta(days=1)
        return _ok(tx, dt, "medium", context="Fim da tarde")
        
    if "tarde" in tx:
        dt = ref.replace(hour=15, minute=0, second=0, microsecond=0)
        if dt < ref:
            dt += timedelta(days=1)
        return _ok(tx, dt, "medium", context="Período da tarde")
        
    if "manhã" in tx or "manha" in tx:
        dt = ref.replace(hour=10, minute=30, second=0, microsecond=0)
        if dt < ref:
            dt += timedelta(days=1)
        return _ok(tx, dt, "medium", context="Período da manhã")

    # ---------------- relative ("em 30 min") ----------------
    rel = RELATIVE_TIME_REGEX.search(tx)
    if rel:
        amt = 0
        if rel.group(1): amt = int(rel.group(1))
        unit = rel.group(2) or ""
        mins = amt if unit.startswith("min") else amt*60
        dt = ref + timedelta(minutes=mins)
        return _ok(tx, dt, "high", is_relative=True)

    return {"original": tx, "interpreted": None,
            "confidence": "low",
            "message": "Não consegui interpretar. Pode confirmar, por favor?"}

def _ok(original: str, dt: _dt, conf: str, is_relative=False, context=""):
    return {
        "original": original,
        "interpreted": dt.strftime("%H:%M"),
        "dt": dt.isoformat(),
        "day_offset": (dt.date() - _dt.now(TZ).date()).days,
        "is_relative": is_relative,
        "confidence": conf,
        "context": context or None,
    } 

# ─────────────────────────  Check Time Availability  ─────────────────────
@function_tool
async def check_time_availability(time_expression: str, date_expression: str = "") -> dict:
    """
    Função unificada que interpreta expressões temporais e verifica disponibilidade.
    
    Esta função combina várias funcionalidades em uma só:
    1. Interpretação de expressões temporais em português (formal e coloquial)
    2. Verificação de disponibilidade nos horários solicitados
    3. Sugestão de alternativas quando o horário não está disponível
    
    Args:
        time_expression: A expressão de tempo como falada pelo cliente
        date_expression: A expressão de data (opcional)
        
    Returns:
        Informações sobre disponibilidade e alternativas
    """
    # Variáveis iniciais
    now = _dt.now(TZ)
    ref_dt = now
    offset_days = 0
    
    # === Processamento de data ===
    if date_expression:
        date_expression = date_expression.lower().strip()
        # Processamento de data relativa simples
        if "amanhã" in date_expression or "amanha" in date_expression:
            ref_dt = now + timedelta(days=1)
            offset_days = 1
        elif "depois de amanhã" in date_expression:
            ref_dt = now + timedelta(days=2)
            offset_days = 2
        # Verificação simples para dias da semana
        elif any(day in date_expression for day in ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]):
            # Para simplicidade, vamos considerar o próximo dia da semana mencionado
            # Em uma implementação mais robusta, isso exigiria um algoritmo mais complexo
            current_day = now.weekday()  # 0 = segunda, 6 = domingo
            target_day = None
            
            if "segunda" in date_expression:
                target_day = 0
            elif "terça" in date_expression or "terca" in date_expression:
                target_day = 1
            elif "quarta" in date_expression:
                target_day = 2
            elif "quinta" in date_expression:
                target_day = 3
            elif "sexta" in date_expression:
                target_day = 4
            elif "sábado" in date_expression or "sabado" in date_expression:
                target_day = 5
            elif "domingo" in date_expression:
                target_day = 6
            
            if target_day is not None:
                # Calcular quantos dias adicionar para chegar ao próximo dia alvo
                days_to_add = (target_day - current_day) % 7
                if days_to_add == 0:  # Se hoje for o dia mencionado, considerar próxima semana
                    days_to_add = 7
                ref_dt = now + timedelta(days=days_to_add)
                offset_days = days_to_add
    
    # === Interpretação do horário solicitado ===
    time_result = await interpret_time_new(time_expression, ref_dt.isoformat())
    
    if not time_result["interpreted"]:
        return {
            "original_expression": time_expression,
            "date_expression": date_expression or "hoje",
            "interpreted_time": None,
            "available": False,
            "reason": "Não consegui entender o horário solicitado. Pode ser mais específico?",
            "status_message": "❌ HORÁRIO NÃO COMPREENDIDO",
            "confidence": "low"
        }
    
    # Converter para datetime completo com a data de referência
    dt_str = time_result["dt"]
    pickup_dt = _dt.fromisoformat(dt_str)
    
    # Considerar o offset de dias calculado para a data de referência
    if offset_days > 0 and time_result.get("day_offset", 0) == 0:
        # Ajustar apenas se interpret_time_new não já tiver ajustado
        pickup_dt = pickup_dt.replace(
            year=ref_dt.year,
            month=ref_dt.month,
            day=ref_dt.day
        )
    
    # === Verificar estado da loja e horário de funcionamento ===
    shop_state = BOOT_DATA["shop_state"] or await compute_shop_state()
    
    # Verificar disponibilidade no mapa de slots carregado do Supabase
    pickup_day = pickup_dt.date()
    pickup_time = pickup_dt.time().strftime("%H:%M")
    
    # Verificar se o restaurante está aberto ou fechado (independente de slots)
    restaurant_status = "ABERTO" if shop_state["status"] == ShopStatus.OPEN.value else "FECHADO"
    
    # Obter slots do dia (do mapa já carregado do Supabase)
    all_slots = BOOT_DATA.get("slot_map", {}).get(pickup_day, [])
    if not all_slots:
        return {
            "original_expression": time_expression,
            "date_expression": date_expression or "hoje",
            "interpreted_time": time_result["interpreted"],
            "interpreted_datetime": dt_str,
            "available": False,
            "status_message": f"❌ DIA INDISPONÍVEL - RESTAURANTE FECHADO",
            "restaurant_status": restaurant_status,
            "reason": "Estamos encerrados nesse dia ou não há horários disponíveis.",
            "confidence": time_result.get("confidence", "medium")
        }
    
    # Verificar slot específico para a hora solicitada
    exact_slot = next((slot for slot in all_slots 
                       if slot.time.strftime("%H:%M") == pickup_time), None)
    
    # Verificar disponibilidade do slot exato
    if exact_slot:
        if exact_slot.available:
            return {
                "original_expression": time_expression,
                "date_expression": date_expression or "hoje",
                "interpreted_time": pickup_time,
                "interpreted_datetime": dt_str,
                "available": True,
                "status_message": f"✅ SLOT DISPONÍVEL - {pickup_time}",
                "restaurant_status": restaurant_status,
                "day_offset": offset_days,
                "context": time_result.get("context"),
                "confidence": time_result.get("confidence", "high")
            }
        else:
            # O slot existe mas não está disponível - buscar alternativas
            available_slots = [s for s in all_slots if s.available]
            
            # Encontrar slot mais próximo
            close_slot = nearest_slot(pickup_dt)
            
            if close_slot:
                # Comparar horários para descritivos naturais
                close_minutes = close_slot.time.hour * 60 + close_slot.time.minute
                pickup_minutes = pickup_dt.hour * 60 + pickup_dt.minute
                minutes_diff = close_minutes - pickup_minutes
                
                time_description = "mais tarde" if minutes_diff > 0 else "mais cedo"
                abs_diff = abs(minutes_diff)
                
                if abs_diff < 15:
                    time_qualifier = "poucos minutos"
                elif abs_diff < 30:
                    time_qualifier = "cerca de meia hora"
                elif abs_diff < 60:
                    time_qualifier = "cerca de uma hora"
                else:
                    time_qualifier = f"{abs_diff // 60} horas"
                
                return {
                    "original_expression": time_expression,
                    "date_expression": date_expression or "hoje",
                    "interpreted_time": pickup_time,
                    "interpreted_datetime": dt_str,
                    "available": False,
                    "status_message": f"❌ SLOT OCUPADO - {pickup_time}",
                    "restaurant_status": restaurant_status,
                    "reason": f"O horário solicitado ({pickup_time}) não está disponível.",
                    "suggested_alternatives": [
                        {"time": close_slot.time.strftime("%H:%M"), 
                         "explanation": f"Temos disponibilidade {time_qualifier} {time_description}, às {close_slot.time.strftime('%H:%M')}"}
                    ],
                    "available_slots": [s.time.strftime("%H:%M") for s in available_slots],
                    "confidence": time_result.get("confidence", "medium")
                }
            
            # Se não encontrou alternativas próximas, listar todos os slots disponíveis
            if available_slots:
                return {
                    "original_expression": time_expression,
                    "date_expression": date_expression or "hoje",
                    "interpreted_time": pickup_time,
                    "interpreted_datetime": dt_str,
                    "available": False,
                    "status_message": f"❌ SLOT OCUPADO - {pickup_time}",
                    "restaurant_status": restaurant_status,
                    "reason": f"O horário solicitado ({pickup_time}) está indisponível.",
                    "suggested_alternatives": [
                        {"time": s.time.strftime("%H:%M"), "explanation": ""} 
                        for s in available_slots[:3]  # Limitar a 3 sugestões
                    ],
                    "available_slots": [s.time.strftime("%H:%M") for s in available_slots],
                    "confidence": time_result.get("confidence", "medium")
                }
            
            # Nenhum slot disponível no dia
            return {
                "original_expression": time_expression,
                "date_expression": date_expression or "hoje",
                "interpreted_time": pickup_time,
                "interpreted_datetime": dt_str,
                "available": False,
                "status_message": f"❌ TODOS SLOTS OCUPADOS",
                "restaurant_status": restaurant_status,
                "reason": f"Não temos mais horários disponíveis nesse dia.",
                "confidence": time_result.get("confidence", "medium")
            }
    
    # Slot exato não encontrado, verificar se está dentro do horário de funcionamento
    # Verificar se o horário está dentro do período de funcionamento
    pickup_minutes = pickup_dt.hour * 60 + pickup_dt.minute
    weekday = pickup_day.strftime("%A").lower()
    schedule_slots = SCHEDULE.get(weekday, [])
    within_hours = any(s.start_minutes <= pickup_minutes < s.end_minutes for s in schedule_slots)
    
    if not within_hours:
        # Formatar os períodos de funcionamento para resposta
        periods = []
        for slot in schedule_slots:
            start = minutes_to_hms(slot.start_minutes)
            end = minutes_to_hms(slot.end_minutes) 
            periods.append(f"{start}-{end}")
        
        readable_hours = ", ".join(periods)
        
        return {
            "original_expression": time_expression,
            "date_expression": date_expression or "hoje",
            "interpreted_time": pickup_time,
            "interpreted_datetime": dt_str,
            "available": False,
            "status_message": f"❌ FORA DO HORÁRIO DE FUNCIONAMENTO",
            "restaurant_status": restaurant_status,
            "reason": f"O horário solicitado está fora do período de funcionamento.",
            "operating_hours": readable_hours,
            "confidence": time_result.get("confidence", "medium")
        }
    
    # Se nenhuma das verificações anteriores retornar, significa que o horário está 
    # dentro do período de funcionamento, mas não há slot específico para ele
    # Sugerir criar o slot
    return {
        "original_expression": time_expression,
        "date_expression": date_expression or "hoje",
        "interpreted_time": pickup_time,
        "interpreted_datetime": dt_str,
        "available": False,
        "status_message": f"❌ SLOT NÃO CADASTRADO",
        "restaurant_status": restaurant_status,
        "reason": f"O horário solicitado ({pickup_time}) não está configurado no sistema.",
        "available_slots": [s.time.strftime("%H:%M") for s in all_slots if s.available],
        "needs_slot_creation": True,
        "confidence": time_result.get("confidence", "medium")
    }

# ─────────────────────────  Order + Transfer  ───────────────────
@function_tool
async def order_confirmed(name: str, pickup_time: str,
                          items: list[str],
                          customizations: dict|None=None):
    log.info("PEDIDO CONFIRMADO — %s %s — %s", name, pickup_time, items)
    
    # Process customizations if available
    customized_items = []
    if customizations:
        for item, options in customizations.items():
            if isinstance(options, dict):
                opts_str = ", ".join(f"{k}: {v}" for k, v in options.items())
                log.info(f"Personalização para {item}: {opts_str}")
                customized_items.append(f"{item} ({opts_str})")
            else:
                log.info(f"Personalização para {item}: {options}")
                customized_items.append(f"{item} ({options})")
    
    # Format the order
    order_header = f"{name} {pickup_time}"
    
    # Combine standard items and customized items
    formatted_items = []
    for item in items:
        # Skip items that are already in customized_items to avoid duplication
        if not any(item in custom_item for custom_item in customized_items):
            formatted_items.append(item)
    
    # Add all customized items
    formatted_items.extend(customized_items)
    
    # Create final transcription
    transcription = order_header + "\n" + "\n".join(formatted_items)
    
    # Extract phone number from room name
    phone_number = "unknown"
    job_ctx = TRANSFER_HUMAN_CONTEXT
    
    if job_ctx and job_ctx.room:
        room_name = job_ctx.room.name
        # Format example: "call-_+351933792547_rgdPNdFqU63Z"
        if room_name and "_" in room_name:
            parts = room_name.split("_")
            if len(parts) >= 2:
                possible_phone = parts[1]
                if possible_phone.startswith("+"):
                    phone_number = possible_phone
                    log.info(f"Extracted phone number for order: {phone_number}")
    
    # Prepare payload with transcription and phone
    payload = {
        "transcription": transcription,
        "phone": phone_number
    }
    
    # Send to webhook
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            "https://hook.eu2.make.com/67puqsnvot28na9cget6444fiejy3go6?type=order-confirmed",
            json=payload, timeout=8) as r:
            response_data = await r.json(content_type=None)
            log.info(f"Order confirmation webhook response: {response_data}")
    
    return {
        "ok": True,
        "message": "Pedido confirmado com sucesso",
        "transcription": transcription,
        "phone": phone_number
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
    "Função: és a operadora telefónica da Churrascaria Quitanda em Portugal. "
    "Fala SEMPRE em Português europeu, usando vocabulário como: 'telemóvel' (não 'celular'), 'pequeno-almoço' (não 'café da manhã'), 'autocarro' (não 'ônibus'). "
    "Sê PROFISSIONAL, EFICIENTE e DIRETO/A. O objetivo é registar os pedidos o mais rapidamente possível. "
    "Evita conversas desnecessárias ou explicações longas. Usa frases curtas e objetivas. "
    
    "DADOS IMPORTANTES (já disponíveis em BOOT_DATA - não uses ferramentas para obtê-los):\n"
    "- Horário de funcionamento: Quando estamos abertos vs. fechados\n"
    "- Slots disponíveis: Horários específicos para reservas/takeaway\n"
    "- Menu: Pratos disponíveis e preços\n\n"
    
    "SISTEMA DE HORÁRIOS E SLOTS:\n"
    "1. Horários de funcionamento: Definem quando o restaurante está aberto ao público.\n"
    "2. Slots de reserva: Horários específicos disponíveis para reservas (slots de 30 minutos).\n"
    "3. IMPORTANTE: Um slot pode existir mas estar marcado como 'indisponível' (available: false) no Supabase.\n"
    "4. Nunca aceites uma reserva para um slot marcado como indisponível, mesmo que esteja dentro do horário normal.\n\n"
    
    "FLUXO EFICIENTE:\n"
    "1. Entende rapidamente o que o cliente quer (takeaway, reserva, informação)\n"
    "2. Verifica disponibilidade no horário pedido usando check_time_availability\n"
    "3. REJEITA SEMPRE reservas para horários indisponíveis ou ocupados\n"
    "4. Confirma os itens do pedido de forma clara e concisa\n"
    "5. Finaliza com confirmação do pedido\n\n"
    
    "Se houver problemas (horário indisponível/item esgotado), oferece alternativas objetivas sem explicações desnecessárias. "
    "Utiliza formas de tratamento como 'o senhor'/'a senhora' ou 'você' em vez de 'tu'.\n\n"
    
    "REGRAS RIGOROSAS PARA DISPONIBILIDADE:\n"
    "- Verifique SEMPRE o campo 'available' (disponível) retornado pela função check_time_availability\n"
    "- Se available=false, NUNCA aceite a reserva - comunique a indisponibilidade e sugira alternativas\n"
    "- Se available=true, pode confirmar a reserva\n"
    "- Sempre ofereça os slots alternativos retornados pela função quando o horário desejado estiver indisponível\n\n"
    
    "NOVA FERRAMENTA INTELIGENTE - check_time_availability:\n\n"
    "Utiliza a ferramenta check_time_availability para verificar disponibilidade de horários. Esta é uma ferramenta "
    "flexível e inteligente que depende de ti para interpretar os resultados e tomar decisões.\n\n"
    "Aqui está o que deves fazer com ela:\n\n"
    "1. Recebe a expressão de tempo do cliente (ex: 'sete e meia', '19h30', 'almoço')\n"
    "2. Chama check_time_availability(time_expression, date_expression)\n"
    "3. Analisa o resultado para:\n"
    "   - Confirmar se o horário foi interpretado corretamente\n"
    "   - Verificar disponibilidade ('available' - true/false)\n"
    "   - Entender a razão para indisponibilidade ('reason')\n"
    "   - Verificar alternativas sugeridas\n\n"
    "4. Responde com naturalidade, explicando ao cliente:\n"
    "   - Se temos disponibilidade (confirmando o horário)\n"
    "   - Se não temos, explica o porquê e sugere alternativas de forma natural\n\n"
    
    "EXEMPLOS CRÍTICOS DE VERIFICAÇÃO DE DISPONIBILIDADE:\n"
    "Cliente: 'Queria reservar para amanhã às 20h'\n"
    "Tu: [usas check_time_availability('20h', 'amanhã')]\n"
    "- Se available=true: 'Perfeito, temos mesa disponível amanhã às 20h.'\n"
    "- Se available=false: 'Lamento, mas já não temos disponibilidade para amanhã às 20h. Temos disponibilidade às 19h30 ou às 20h30, se preferir.'\n"
    "- Se fechado: 'Lamento informar que estamos encerrados amanhã. O nosso próximo dia de funcionamento é sexta-feira.'\n\n"
    
    "Esta ferramenta é flexível para que possas adaptá-la a diferentes situações, interpretando os dados e comunicando de forma natural com o cliente."
)

def build_system_prompt() -> str:
    """Gera o prompt do sistema com dados do BOOT_DATA + nudge PT-PT."""
    now  = _dt.now(TZ)
    dia  = now.strftime("%A")
    hora = now.strftime("%H:%M")

    shop = BOOT_DATA["shop_state"] or {}
    menu = BOOT_DATA["menu"]       or "Menu indisponível"

    # Obter slots disponíveis para hoje usando a nova funcionalidade
    today = now.date()
    today_slots = get_slots_for_day(today)
    slots_txt = ", ".join(s.time.strftime("%H:%M") for s in today_slots) if today_slots else "Nenhum slot disponível hoje"
    
    estado_txt = (
        "ABERTO até " + shop.get("next_close")
        if shop.get("status") == ShopStatus.OPEN.value
        else shop.get("message", "FECHADO")
    )

    # Inclui o bloco de nudging logo após o BASE_PROMPT
    base_prompt_with_nudge = (
        f"{BASE_PROMPT}\n\n{PT_PT_NUDGE_BLOCK}"
    )
    
    # Adicionar instruções específicas sobre ferramentas de verificação e atualização de slots
    tools_instructions = (
        "\n\nFERRAMENTAS ESSENCIAIS PARA GESTÃO DE SLOTS:\n"
        "1. check_time_availability(time_expression, date_expression) - Verificar se um horário está disponível\n"
        "2. update_time_slot_availability(slot_date, slot_time, available) - Criar ou atualizar disponibilidade de um slot\n\n"
        "INSTRUÇÕES IMPORTANTES:\n"
        "- Se um cliente solicitar um horário que não existe como slot, mas está dentro do horário de funcionamento,\n"
        "  NUNCA aceite automaticamente - sugira os slots disponíveis mais próximos no mesmo dia\n"
        "- Slots marcados como available=false estão INDISPONÍVEIS para reserva - nunca os ofereça aos clientes\n"
        "- Dê grande destaque aos campos 'available' e 'status_message' em todas as verificações de disponibilidade\n"
        "- Verifique SEMPRE a disponibilidade exata antes de confirmar uma reserva"
    )

    return (
        f"HORA ATUAL: {hora} ({dia})\n"
        f"ESTADO DO RESTAURANTE: {estado_txt}\n"
        f"HORÁRIO DE FUNCIONAMENTO HOJE: {shop.get('today_readable_hours','')}\n"
        f"SLOTS DISPONÍVEIS PARA RESERVAS HOJE: {slots_txt}\n\n"
        f"{base_prompt_with_nudge}\n\n"
        f"{tools_instructions}\n\n"
        f"MENU:\n{menu}\n\n"
        "⚠️ IMPORTANTE: Todos os dados necessários já estão carregados no BOOT_DATA - usa-os diretamente em vez de chamar ferramentas.\n"
        "⚠️ EFICIÊNCIA: O objetivo é registar pedidos rapidamente. Evita explicações desnecessárias.\n"
        "⚠️ TRATAMENTO: Usa 'o senhor/a senhora' ou 'você' para maior formalidade e profissionalismo.\n"
        "⚠️ PRECISÃO: Distingue entre horário de funcionamento e slots disponíveis para reservas."
    )

# ─────────────────────────  Entrypoint  ─────────────────────────
TRANSFER_HUMAN_CONTEXT: JobContext|None = None   # para função acima

async def entrypoint(ctx: JobContext):
    global TRANSFER_HUMAN_CONTEXT
    TRANSFER_HUMAN_CONTEXT = ctx

    await boot_fetch_menu()

    llm = openai.realtime.RealtimeModel(
        model="gpt-4o-realtime-preview", voice="coral",
        temperature=0.7,
        turn_detection=TurnDetection(
            type="semantic_vad", eagerness="auto",
            create_response=True, interrupt_response=True)
    )

    tools = [get_menu, get_shop_state,
             validate_pickup, validate_pickup_combined,
             interpret_time, order_confirmed, transfer_human,
             check_time_availability, refresh_data_from_supabase,
             update_time_slot_availability]  # Adding the new time slot tool

    agent = Agent(instructions=build_system_prompt(), tools=tools)
    session = AgentSession(llm=llm)

    await ctx.connect()
    await session.start(agent, room=ctx.room)

    await session.generate_reply(
        instructions="Churrascaria Quitanda, boa tarde. Em que posso ajudá-lo?"
    )

# ─────────────────────────  Runner  ────────────────────────────
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint,
                              worker_type=WorkerType.ROOM))

def parse_hoursanddate(raw_text: str, today: date) -> dict[date, list[PickupSlot]]:
    """
    Converte a string do Make num mapa {date: [PickupSlot,…]}.
    """
    slot_map: dict[date, list[PickupSlot]] = {}
    for m in SLOT_REGEX.finditer(raw_text):
        d_str = m.group("date") or today.isoformat()
        d = _dt.fromisoformat(d_str).date()
        t = _dt.strptime(m.group("time"), "%H:%M").time()
        avail = m.group("disp").lower() == "sim"
        slot_map.setdefault(d, []).append(PickupSlot(d, t, avail))
    # ordena por hora dentro de cada dia
    for sl in slot_map.values():
        sl.sort(key=lambda s: (s.time.hour, s.time.minute))
    return slot_map 

async def fetch_time_slots() -> dict[date, list[PickupSlot]]:
    """
    Fetch time slots directly from Supabase's time_slots table
    
    Returns:
        Dictionary mapping dates to PickupSlot objects
    """
    if not supabase:
        log.warning("Supabase não está configurado. Gerando slots a partir dos horários padrão.")
        today = _dt.now(TZ).date()
        return generate_slot_map(SCHEDULE, today)
    
    try:
        # Query time slots for the next 7 days
        today = _dt.now(TZ).date()
        one_week_later = today + timedelta(days=7)
        
        # Fix: Using the correct order() syntax (column name only, no direction parameter)
        response = supabase.table("time_slots").select("*")\
            .gte("slot_date", today.isoformat())\
            .lt("slot_date", one_week_later.isoformat())\
            .order("slot_date")\
            .execute()
            
        slots_data = response.data
        
        if not slots_data:
            log.warning("Não foram encontrados slots na base de dados. Gerando a partir dos horários.")
            return generate_slot_map(SCHEDULE, today)
        
        # Convert to the format expected by the application
        slot_map = {}
        for record in slots_data:
            slot_date = _dt.fromisoformat(record["slot_date"]).date()
            slot_time_str = record["slot_time"]
            h, m = map(int, slot_time_str.split(":")[0:2])
            slot_time = time(hour=h, minute=m)
            available = record["available"]
            
            if slot_date not in slot_map:
                slot_map[slot_date] = []
                
            slot_map[slot_date].append(PickupSlot(
                date=slot_date,
                time=slot_time,
                available=available
            ))
        
        # Sort slots by time within each day
        for day_slots in slot_map.values():
            day_slots.sort(key=lambda s: (s.time.hour, s.time.minute))
            
        log.info(f"Slots carregados do Supabase: {sum(len(slots) for slots in slot_map.values())} slots em {len(slot_map)} dias")
        return slot_map
    except Exception as e:
        log.error(f"Erro ao buscar slots do Supabase: {e}")
        today = _dt.now(TZ).date()
        return generate_slot_map(SCHEDULE, today)

@function_tool
async def update_time_slot_availability(slot_date: str, slot_time: str, available: bool) -> dict:
    """
    Update the availability of a specific time slot or create a new slot if it doesn't exist.
    This is useful for:
    - Configurar um novo horário de reserva
    - Marcar um horário como disponível/indisponível
    - Criar horários de reserva para dias especiais
    
    Args:
        slot_date: Data no formato ISO (AAAA-MM-DD), exemplo: "2025-05-10"
        slot_time: Hora no formato 24 horas (HH:MM), exemplo: "19:30"
        available: Se o slot está disponível (true) ou ocupado (false)
        
    Returns:
        Dictionary with status of the update and a clear message
    """
    if not supabase:
        return {
            "success": False,
            "message": "Supabase não está configurado. Não é possível atualizar slots.",
            "status_message": "❌ ERRO DE CONFIGURAÇÃO"
        }
    
    try:
        # Format time for database query (ensure HH:MM format)
        if ":" not in slot_time:
            if len(slot_time) == 4:  # Format like 1930
                slot_time = f"{slot_time[:2]}:{slot_time[2:]}"
            else:
                return {
                    "success": False,
                    "message": f"Formato de hora inválido: {slot_time}. Use HH:MM.",
                    "status_message": "❌ FORMATO DE HORA INVÁLIDO"
                }
        
        # Ensure time has seconds for PostgreSQL TIME format
        if len(slot_time.split(":")) == 2:
            slot_time = f"{slot_time}:00"
            
        # Check if the slot exists
        response = supabase.table("time_slots")\
            .select("id")\
            .eq("slot_date", slot_date)\
            .eq("slot_time", slot_time)\
            .execute()
            
        if not response.data:
            # Create the slot if it doesn't exist
            insert_response = supabase.table("time_slots").insert({
                "slot_date": slot_date,
                "slot_time": slot_time,
                "available": available,
                "capacity": 4  # Default capacity
            }).execute()
            
            if insert_response.data:
                # Refresh the slot map
                today = _dt.now(TZ).date()
                BOOT_DATA["slot_map"] = await fetch_time_slots()
                
                status = "DISPONÍVEL" if available else "OCUPADO"
                return {
                    "success": True,
                    "message": f"Slot criado para {slot_date} às {slot_time[:5]} com disponibilidade: {status}.",
                    "status_message": f"✅ SLOT CRIADO - {status}",
                    "action": "created",
                    "available": available
                }
            else:
                return {
                    "success": False,
                    "message": f"Erro ao criar slot para {slot_date} às {slot_time[:5]}.",
                    "status_message": "❌ ERRO AO CRIAR SLOT",
                    "available": False
                }
        else:
            # Update the existing slot
            slot_id = response.data[0]["id"]
            update_response = supabase.table("time_slots")\
                .update({"available": available, "updated_at": _dt.now(TZ).isoformat()})\
                .eq("id", slot_id)\
                .execute()
                
            if update_response.data:
                # Refresh the slot map
                today = _dt.now(TZ).date()
                BOOT_DATA["slot_map"] = await fetch_time_slots()
                
                status = "DISPONÍVEL" if available else "OCUPADO"
                return {
                    "success": True,
                    "message": f"Disponibilidade do slot para {slot_date} às {slot_time[:5]} atualizada para {status}.",
                    "status_message": f"✅ SLOT ATUALIZADO - {status}",
                    "action": "updated",
                    "available": available
                }
            else:
                return {
                    "success": False,
                    "message": f"Erro ao atualizar slot para {slot_date} às {slot_time[:5]}.",
                    "status_message": "❌ ERRO AO ATUALIZAR SLOT",
                    "available": False
                }
    except Exception as e:
        log.error(f"Erro ao atualizar disponibilidade do slot: {e}")
        return {
            "success": False,
            "message": f"Erro: {str(e)}",
            "status_message": "❌ ERRO DE SISTEMA"
        } 