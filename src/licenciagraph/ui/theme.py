"""Identidade visual: verdes e azuis da SUDEMA, barra lateral escura e layout de chat."""

from __future__ import annotations

import base64
from functools import cache
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"
LOGO = ASSETS / "sudema-icone.png"

BLUE = "#0B5CA8"
GREEN = "#1F8A4C"
NAVY = "#0A2A4A"
FOREST = "#0B3D34"

ICONS = ("plus", "close", "chat", "assistant", "search", "about", "settings")


@cache
def icon_uri(name: str) -> str:
    data = (ASSETS / "icons" / f"{name}.svg").read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode()


def _icon_rule(selector: str, name: str, size: int = 18) -> str:
    return (
        f"{selector} button::before {{ content: ''; flex: none; width: {size}px; "
        f"height: {size}px; background: url({icon_uri(name)}) center / contain "
        "no-repeat; opacity: 0.9; }"
    )


CSS = f"""
<style>
#MainMenu, footer, .stAppDeployButton {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 52rem; padding-top: 2rem; padding-bottom: 7rem; }}
h1, h2, h3, h4 {{ color: {BLUE}; letter-spacing: -0.01em; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY} 0%, #0B3550 55%, {FOREST} 100%);
    border-right: 0;
}}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding: 0.4rem 0.2rem; }}
[data-testid="stSidebar"] * {{ color: #E6F0F7; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.25rem; }}
[data-testid="stSidebarHeader"] {{ padding: 0.8rem 1rem 0; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.12); margin: 0.6rem 0; }}

.brand {{ display: flex; align-items: center; gap: 0.75rem; padding: 0.2rem 0.5rem 0.9rem; }}
.brand img {{ width: 42px; height: 42px; border-radius: 50%; background: #fff; padding: 2px; }}
.brand .name {{ font-size: 1.45rem; font-weight: 800; letter-spacing: 0.04em; color: #fff;
    line-height: 1; }}
.brand .sub {{ font-size: 0.7rem; letter-spacing: 0.05em; text-transform: uppercase;
    color: #9fd8b6; margin-top: 0.25rem; }}
.side-label {{ font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: #8fb3cc; padding: 0.7rem 0.6rem 0.2rem; }}
.side-note {{ font-size: 0.72rem; line-height: 1.35; color: #9db8cb; padding: 0.4rem 0.6rem; }}
.status {{ font-size: 0.74rem; padding: 0.1rem 0.6rem; color: #9db8cb; }}
.status i {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 6px; background: #3ddc84; }}
.status i.off {{ background: #ff6b6b; }}

[data-testid="stSidebar"] .stButton > button {{
    display: flex; align-items: center; justify-content: flex-start; gap: 0.65rem;
    width: 100%; background: transparent; border: 0; border-radius: 10px;
    padding: 0.5rem 0.7rem; min-height: 2.3rem; font-size: 0.9rem; text-align: left;
    box-shadow: none; transition: background 0.15s;
}}
[data-testid="stSidebar"] .stButton > button > div {{ flex: 1; justify-content: flex-start; }}
[data-testid="stSidebar"] .stButton > button p {{
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: left;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,255,255,0.09);
}}
[data-testid="stSidebar"] .stButton > button:focus:not(:active) {{
    border: 0; box-shadow: none;
}}
[class*="st-key-del_"] button {{
    opacity: 0; justify-content: center; padding: 0.4rem; min-height: 2.3rem;
}}
[data-testid="stHorizontalBlock"]:hover [class*="st-key-del_"] button {{ opacity: 0.7; }}
[class*="st-key-del_"] button:hover {{ opacity: 1; background: rgba(255,107,107,0.25); }}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{ gap: 0.1rem; align-items: center; }}

.st-key-new_chat button > div {{ justify-content: center; }}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {{
    min-height: calc(100vh - 6rem);
}}
[data-testid="stLayoutWrapper"]:has(> .st-key-sidebar_footer) {{ margin-top: auto; }}
.st-key-new_chat button {{
    background: linear-gradient(135deg, {GREEN}, #17a2a0 120%) !important;
    font-weight: 600; justify-content: center !important; padding: 0.7rem 0.9rem !important;
    min-height: 2.9rem !important; border-radius: 14px !important;
    box-shadow: 0 6px 16px rgba(31,138,76,0.35) !important;
}}
.st-key-new_chat button:hover {{
    filter: brightness(1.08); transform: translateY(-1px);
}}

.hero {{ text-align: center; margin: 3.2rem 0 1.6rem; }}
.hero h1 {{ font-size: 2rem; margin-bottom: 0.4rem; }}
.hero p {{ color: #47657a; font-size: 1.02rem; margin: 0 auto; max-width: 40rem; }}

.main .stButton > button {{
    text-align: left; justify-content: flex-start; border-radius: 14px;
    border: 1px solid #cfe3d8; background: #F6FBF8; color: {BLUE};
    padding: 0.8rem 1rem; min-height: 4.2rem; font-size: 0.93rem;
}}
.main .stButton > button p {{ white-space: normal; text-align: left; }}
.main .stButton > button:hover {{
    border-color: {GREEN}; background: #E9F6EE; color: {BLUE};
}}

[data-testid="stChatMessage"] {{
    background: transparent; padding: 0.4rem 0; gap: 0.75rem;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
    flex-direction: row-reverse; text-align: left;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {{
    background: #E1F1E8; border-radius: 20px 20px 4px 20px;
    padding: 0.7rem 1.05rem; max-width: 80%; width: fit-content;
}}
[data-testid="stChatMessageAvatarUser"] {{ display: none; }}
[data-testid="stChatMessageAvatarAssistant"] {{
    background: linear-gradient(135deg, {GREEN}, {BLUE}); color: white;
}}

[data-testid="stChatInput"] {{
    border: 1px solid #b9d8c7; border-radius: 26px;
    box-shadow: 0 4px 18px rgba(11, 92, 168, 0.10);
}}
[data-testid="stBottom"] > div {{
    background: linear-gradient(0deg, #fff 70%, transparent);
}}

[data-testid="stExpander"] {{
    border: 1px solid #d7e6dd; border-radius: 12px; background: #FBFDFC;
}}
.chip {{
    display: inline-block; background: #E2EEF8; color: {BLUE};
    border-radius: 14px; padding: 0.3rem 0.8rem; font-size: 0.82rem; margin: 0.2rem 0;
}}
.note {{ color: #5b7488; font-size: 0.78rem; }}

{_icon_rule(".st-key-new_chat", "plus")}
{_icon_rule(".st-key-nav_chat", "chat")}
{_icon_rule(".st-key-nav_guide", "assistant")}
{_icon_rule(".st-key-open_settings", "settings")}
{_icon_rule('[class*="st-key-del_"]', "close", 14)}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def active_css(active_keys: list[str]) -> None:
    """Destaca, na barra lateral, a página e a conversa ativas."""
    rules = "".join(
        f".st-key-{key} button {{ background: rgba(255,255,255,0.16) !important; }}"
        for key in active_keys
    )
    if rules:
        st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)


def render_brand() -> None:
    """Logo e nome da SUDEMA no topo da barra lateral."""
    logo = ""
    if LOGO.exists():
        data = base64.b64encode(LOGO.read_bytes()).decode()
        logo = f'<img src="data:image/png;base64,{data}" alt="SUDEMA"/>'
    st.markdown(
        f'<div class="brand">{logo}<div><div class="name">SUDEMA</div>'
        '<div class="sub">Licenciamento ambiental · PB</div></div></div>',
        unsafe_allow_html=True,
    )
