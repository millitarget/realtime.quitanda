# Churrascaria Quitanda - Agente de Atendimento Telefônico

Um agente virtual baseado em LiveKit e OpenAI para atendimento telefônico da Churrascaria Quitanda, desenvolvido em Python.

## Funcionalidades

- Atendimento automatizado em Português de Portugal
- Processamento de encomendas para take-away
- Reconhecimento e validação de horários disponíveis
- Integração com o sistema de menu e disponibilidade de slots
- Confirmação de pedidos

## Requisitos

- Python 3.10+
- LiveKit
- OpenAI API
- tzdata (para suporte a fusos horários no Windows)

## Configuração

1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Crie um arquivo `.env.local` com suas configurações:

```
# OpenAI
OPENAI_API_KEY=sua_chave_aqui

# LiveKit
LIVEKIT_URL=sua_url_livekit
LIVEKIT_API_KEY=sua_api_key
LIVEKIT_API_SECRET=seu_secret

# Configurações do app
APP_TIMEZONE=Europe/Lisbon
MENU_WEBHOOK_URL=url_do_webhook_menu
MENU_CACHE_TTL=2
```

## Arquivos Principais

- `main.py` - Versão completa do agente com todas as funcionalidades
- `main2.py` - Versão simplificada do agente

## Iniciando o Agente

```bash
python main.py
```

## Estrutura do Projeto

O agente funciona com base em um sistema de turnos (turn detection) e gerenciamento de estado que:

1. Verifica o estado atual da loja (aberto/fechado) baseado no horário
2. Processa os pedidos e valida os horários de retirada
3. Confirma os pedidos e transfere para um operador humano quando necessário

O fluxo de processamento segue as regras de negócio da Churrascaria Quitanda, incluindo horários oficiais de funcionamento e opções do menu. 