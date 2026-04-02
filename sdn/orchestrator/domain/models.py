"""
Modelos de dados da API REST (contratos de entrada).

Separados da lógica de negócio seguindo o princípio de Responsabilidade
Única (SRP): este módulo existe apenas para definir os schemas de entrada
da API, sem nenhuma lógica de processamento.
"""

from pydantic import BaseModel


class SwitchRequest(BaseModel):
    """Payload para bloquear/desbloquear um switch via /manage/switch."""
    switch_id: str
    action: str   # "block" | "unblock"


class IPBlockRequest(BaseModel):
    """Payload para bloquear/desbloquear um IP via /manage/ip."""
    ip: str
    action: str   # "block" | "unblock"
