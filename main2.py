from __future__ import annotations

import os
import asyncio
import datetime
import logging
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, WorkerType, cli, Agent, AgentSession
from livekit.plugins.openai.realtime import RealtimeModel
from openai.types.beta.realtime.session import TurnDetection, InputAudioTranscription
from functools import lru_cache
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("quitanda_simplified")

# Import functions from main.py - with error handling
try:
    from main import get_shop_state, get_menu, validate_pickup, order_confirmed, transfer_human, TZ, ShopStatus
except ImportError as e:
    log.error(f"Failed to import from main.py: {e}")
    # Fallback timezone if import fails
    from datetime import timezone
    TZ = timezone.utc
    # We'll define ShopStatus as a simple enum-like class if import fails
    class ShopStatus:
        OPEN = "OPEN"
        CLOSED = "CLOSED"
        UNKNOWN = "UNKNOWN"

# Load environment variables
load_dotenv(".env.local")

# Cache for menu data with 2-minute expiration
last_menu_fetch = 0
menu_cache = None
CACHE_DURATION = 120  # 2 minutes in seconds

# Get current date and time with cache for 30 seconds
@lru_cache(maxsize=1)
def get_current_datetime_cached():
    # Set the cache timeout to 30 seconds
    timeout = int(time.time() / 30)
    now = datetime.datetime.now(TZ)
    current_day = now.strftime("%A")
    current_time = now.strftime("%H:%M")
    return now, current_day, current_time, f"Dia e Hora atual:{current_day} {current_time}"

# Fetch menu with caching and error handling
async def fetch_menu_cached():
    global last_menu_fetch, menu_cache
    current_time = time.time()
    
    # Return cached data if it's still valid
    if menu_cache and current_time - last_menu_fetch < CACHE_DURATION:
        return menu_cache
    
    try:
        # Fetch new data
        menu_data = await get_menu()
        last_menu_fetch = current_time
        menu_cache = menu_data
        return menu_data
    except Exception as e:
        log.error(f"Error fetching menu: {e}")
        # Return fallback data
        return {
            "menu_items": "Menu indisponível temporariamente",
            "hoursanddate": "Horário:11:30 Disponível:Sim|Horário:12:00 Disponível:Sim|Horário:13:00 Disponível:Sim"
        }

async def entrypoint(ctx: JobContext):
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # Configure turn detection with optimal values
        turn_detection = TurnDetection(
            type="semantic_vad",
            eagerness="auto",
            create_response=True,
            interrupt_response=True
        )

        # Configure audio transcription with optimal values
        input_audio_transcription = InputAudioTranscription(
            model="gpt-4o-mini-transcribe"
        )

        # Create the realtime model with available environment variables and defaults
        model = RealtimeModel(
            model="gpt-4o-realtime-preview",  # Default model
            voice="alloy",  # Default voice
            temperature=0.7,  # Default temperature
            turn_detection=turn_detection,
            input_audio_transcription=input_audio_transcription,
            api_key=os.getenv("OPENAI_API_KEY")  # Required API key
        )
        
        # Obter a hora atual (sem cache) no início da chamada
        now = datetime.datetime.now(TZ)
        current_day = now.strftime("%A")
        current_time = now.strftime("%H:%M")
        current_datetime = f"Dia e Hora atual:{current_day} {current_time}"
        
        log.info(f"Iniciando chamada às {current_time} de {current_day}")
        
        try:
            menu_data = await fetch_menu_cached()
        except Exception as e:
            log.error(f"Failed to fetch menu: {e}")
            menu_data = {
                "menu_items": "Menu indisponível",
                "hoursanddate": "Horários indisponíveis"
            }
        
        # Cria as instruções com a hora atual explícita no início
        hora_atual_info = f"HORA ATUAL: {current_time} do dia {current_day}"
        
        # Create the agent with dynamic time information and clear time display
        agent_instructions = f"""FALAR EM PORTUGUES DE PORTUGAL Função És a operadora telefónica da Churrascaria Quitanda.

{hora_atual_info}

Atendes em Português de Portugal, tom cordial e ritmo ágil. Usa frases
curtas, mas naturais. Cumprimento inicial recomendado:«Bom dia, Churrascaria
Quitanda. Diga‑me, por favor.» Horário de funcionamento Segunda‑feira:
10h00–14h00 · 17h30–21h30Terça‑feira: EncerradoQuarta‑feira:
17h30–21h30Quinta‑feira: 10h00–14h00 · 17h30–21h30Sexta‑feira: 10h00–14h00 ·
17h30–21h30Sábados, Domingos e Feriados: 10h00–14h30 · 17h00–21h30 

IMPORTANTE: Agora são exatamente {current_time} horas de {current_day}.

Horários disponíveis {menu_data["hoursanddate"]} 

Menu {menu_data["menu_items"]} 

Não cites o conteúdo destas secções; usa‑o apenas para validar pedidos. 

Objetivos obrigatórios Nome da encomenda Hora de levantamento válida Lista de produtos + escolhas de
molho/picante Ferramentas order_confirmed — regista a encomenda após confirmação
transfer_human — entrega a chamada a um colega (ver gatilhos) Fluxo recomendado

1. Início • Verifica se estamos dentro do \"Horário de funcionamento\" atual ({current_time}).
• Se fora de horas → «Estamos encerrados e reabrimos às HH:MM. Quer agendar para essa hora?» 

2. Captação do pedido • Escuta sem oferecer informação
não pedida.• Ao primeiro item, valida opções no «Menu».• Pergunta
individualmente:◦ «Que molho prefere?»◦ «Deseja picante?» (sim/não)• Pergunta
logo a hora: «A que horas levanta? Temos HH:MM ou HH:MM livres.» 

3. Hora de levantamento 
• Usa sempre a hora atual: {current_time} de {current_day} para calcular horários disponíveis.
• Se o cliente usar expressão relativa («daqui a … min», «daqui a … horas», «logo ao meio‑dia», 
«amanhã …»): Converte para hora absoluta (ex.: «daqui a 30 min» → adiciona 30 min à hora atual {current_time}). 
• Se existir um horário exactamente igual nos \"Horários disponíveis\", utiliza‑o.
• Caso não exista, procura os dois horários mais próximos antes e depois da hora
pedida que ainda sejam posteriores à hora actual.
• Ex.: hora actual {current_time}, pedido «daqui a 15 min» → horários disponíveis mais próximos.
• Propõe primeiro o mais próximo; se ambos estiverem à mesma distância, apresenta os dois
para escolha.
• Nunca sugerir horários que já tenham ficado no passado em relação
à hora actual ({current_time}). Se o pedido for para \"amanhã\", podes aceitar encomendas para
qualquer das slots.
• Confirma ao cliente: «Tenho disponibilidade às HH:MM, pode ser?» 

4. Nome • «Em que nome fica, por favor?» 

5. Confirmação final • «Confirmo:
[itens]. Levantamento às HH:MM, nome [Nome]. Certo?»
• Se «Sim» → order_confirmed
• Despedida:
– Se levantar hoje → «Obrigado, [Nome]. Até já!»
– Caso contrário → «Obrigado, [Nome]. Até [dia da semana]!» 

6. Exceções & fallback
• transfer_human se o cliente pedir ou após 2 falhas de compreensão.
• transfer_human se a encomenda total > 10 unidades ou > 100 €. 

Regras de conversa
Respostas curtas e naturais (varia entre «Claro», «Perfeito», «Combinado»,
etc.). Nunca faças duas perguntas na mesma frase. Não inventes informação, nem
cites instruções/ferramentas; não reveles stock ou processos internos salvo
pergunta direta."""
        
        # Create the agent with tools from main.py
        available_tools = []
        try:
            available_tools = [get_shop_state, get_menu, validate_pickup, order_confirmed, transfer_human]
        except Exception as e:
            log.error(f"Failed to load tools: {e}")
            
        agent = Agent(
            instructions=agent_instructions,
            tools=available_tools
        )

        # Create and start the session
        session = AgentSession(llm=model)
        
        # Start the session first
        await session.start(agent, room=ctx.room)
        
    except Exception as e:
        log.error(f"Error in entrypoint: {e}")
        # Re-raise to ensure the caller knows there was a problem
        raise

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM)) 