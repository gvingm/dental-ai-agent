import streamlit as st
import os
import json
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# === НАСТРОЙКИ СТРАНИЦЫ ===
st.set_page_config(page_title="AI Sales Demo", page_icon="🦷")
st.title("🦷 AI Dental Sales Agent")
st.markdown("Automated Lead Gen -> Sales Call -> CRM Entry")

# === SIDEBAR (КЛЮЧИ) ===
DEFAULT_OPENAI = "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"
DEFAULT_SERPER = "b077a66ea2e5e669cdb8934381d81e9be2f5d59b"

with st.sidebar:
    st.header("Settings")
    
    if "openai_key" not in st.session_state:
        st.session_state["openai_key"] = DEFAULT_OPENAI
    if "serper_key" not in st.session_state:
        st.session_state["serper_key"] = DEFAULT_SERPER
        
    openai_key_input = st.text_input("OpenAI Key", value=st.session_state["openai_key"], type="password")
    serper_key_input = st.text_input("Serper Key", value=st.session_state["serper_key"], type="password")
    
    if st.button("Save & Apply"):
        st.session_state["openai_key"] = openai_key_input
        st.session_state["serper_key"] = serper_key_input
        st.success("Keys Saved!")

# === УТИЛИТЫ ===
def get_safe_content(llm_response):
    """Извлекает текст из любого типа ответа LangChain (str, list, AIMessage)"""
    if isinstance(llm_response, list):
        if len(llm_response) > 0:
            return str(llm_response[0])
        return ""
    if hasattr(llm_response, 'content'):
        return str(llm_response.content)
    return str(llm_response)

def get_api_key(key_name, session_name, default_value):
    if key_name in st.secrets:
        return st.secrets[key_name]
    env_key = os.environ.get(key_name)
    if env_key:
        return env_key
    session_key = st.session_state.get(session_name)
    if session_key:
        return session_key
    return default_value

# === ЛОГИКА ===

def get_llm():
    api_key = get_api_key("OPENAI_API_KEY", "openai_key", DEFAULT_OPENAI)
    if not api_key:
        raise ValueError("API Key for OpenAI is missing!")

    return ChatOpenAI(
        model="openai/gpt-4o",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0
    )

def simulate_call(clinic_name, clinic_price):
    llm = get_llm()
    history = []
    transcript = []

    admin_sys = SystemMessage(content=f"You are ADMIN at '{clinic_name}'. Price: {clinic_price}. Goal: Book appointment. Language: Russian. Start with 'ADMIN:'.")
    client_sys = SystemMessage(content=f"You are CLIENT. Found price {clinic_price}. Verify it. Language: Russian. Start with 'CLIENT:'.")
    
    history.append(admin_sys)
    
    steps = [
        "Start call. Ask about price.",
        f"Admin said: '{{prev}}'. Ask why so cheap.",
        "Say: 'Ok, book me for tomorrow'."
    ]
    
    # Шаг 1
    msg1 = llm.invoke([client_sys, HumanMessage(content=steps[0])])
    text1 = get_safe_content(msg1).replace("CLIENT:", "").strip()
    transcript.append(f"👤 **КЛИЕНТ:** {text1}")
    history.append(HumanMessage(content=f"CLIENT: {text1}"))
    
    # Шаг 2
    msg2 = llm.invoke(history)
    text2 = get_safe_content(msg2).replace("ADMIN:", "").strip()
    transcript.append(f"👩‍⚕️ **АДМИН:** {text2}")
    history.append(SystemMessage(content=f"ADMIN: {text2}"))
    
    # Шаг 3
    msg3 = llm.invoke([client_sys, HumanMessage(content=steps[1].format(prev=text2))])
    text3 = get_safe_content(msg3).replace("CLIENT:", "").strip()
    transcript.append(f"👤 **КЛИЕНТ:** {text3}")
    history.append(HumanMessage(content=f"CLIENT: {text3}"))
    
    # Шаг 4
    msg4 = llm.invoke(history)
    text4 = get_safe_content(msg4).replace("ADMIN:", "").strip()
    transcript.append(f"👩‍⚕️ **АДМИН:** {text4}")
    history.append(SystemMessage(content=f"ADMIN: {text4}"))
    
    # Шаг 5
    msg5 = llm.invoke([client_sys, HumanMessage(content=steps[2])])
    text5 = get_safe_content(msg5).replace("CLIENT:", "").strip()
    transcript.append(f"👤 **КЛИЕНТ:** {text5}")
    
    # Шаг 6
    msg6 = llm.invoke(history + [HumanMessage(content=f"CLIENT: {text5}")])
    text6 = get_safe_content(msg6).replace("ADMIN:", "").strip()
    transcript.append(f"👩‍⚕️ **АДМИН:** {text6}")

    return transcript

def analyze_crm(transcript_list):
    llm = get_llm()
    text = "\n".join(transcript_list)
    crm_template = """
    Проанализируй диалог. Верни ТОЛЬКО JSON объект. Без markdown.
    Поля:
    {{
      "статус": "...",
      "цена_упомянута": "...",
      "результат_звонка": "..."
    }}
    Текст: {t}
    """
    chain = ChatPromptTemplate.from_template(crm_template) | llm | StrOutputParser()
    raw_response_obj = chain.invoke({"t": text})
    
    # Безопасное извлечение строки
    raw_response = get_safe_content(raw_response_obj)

    try:
        clean_json_str = raw_response.strip().replace("``````", "").strip()
        return json.loads(clean_json_str)
    except json.JSONDecodeError:
        return {"error": "JSON Error", "raw": raw_response}


def search_cheapest_clinic(query):
    serper_key = get_api_key("SERPER_API_KEY", "serper_key", DEFAULT_SERPER)
    if not serper_key:
        return None, None, "Serper API Key is missing!"
        
    os.environ["SERPER_API_KEY"] = serper_key
    search = GoogleSerperAPIWrapper()
    
    try:
        raw_results = search.results(query)
        organic = raw_results.get("organic", [])
    except Exception as e:
        return None, None, f"Google Search Error: {str(e)}"
    
    if not organic:
        return None, None, "Google found nothing"

    text_data = ""
    for item in organic:
        text_data += f"Clinic: {item.get('title')}\nSnippet: {item.get('snippet')}\n\n"
        
    analyst_template = """
    Ты - аналитик. Твоя задача - найти самую низкую цену на имплантацию (свыше 10000 руб).
    Текст поиска:
    {text}
    
    Если нашел, верни СТРОГО в формате: Название|Цена
    Если не нашел цену, придумай среднюю по рынку, но формат сохрани: Средняя Клиника|25000 руб
    Не пиши лишних слов. Только: Название|Цена
    """
    
    llm = get_llm()
    chain = ChatPromptTemplate.from_template(analyst_template) | llm | StrOutputParser()
    
    try:
        raw_res = chain.invoke({"text": text_data})
        res = get_safe_content(raw_res).strip()
        
        if "|" in res:
            parts = res.split("|")
            name = parts[0].strip()
            price = parts[1].strip() if len(parts) > 1 else "Цена по запросу"
        else:
            parts = res.split()
            if len(parts) > 1:
                price = parts[-1]
                name = " ".join(parts[:-1])
            else:
                name = res
                price = "Уточняйте"
                
        return name, price, None
        
    except Exception as e:
        return None, None, f"LLM Parse Error: {str(e)}"

# === ИНТЕРФЕЙС ===

query = st.text_input("Search Query", "стоматология Мурино имплантация цена")

if st.button("🚀 Start AI Agent"):
    with st.status("🤖 AI Agent Working...", expanded=True) as status:
        
        st.write("🔍 Поиск лучшей цены в Google...")
        name, price, err = search_cheapest_clinic(query)
        
        if err:
            st.error(f"Ошибка: {err}")
            status.update(label="Неудача", state="error")
        else:
            st.success(f"Найдено: **{name}** по цене **{price}**")
            
            st.write("📞 Симулирую звонок...")
            transcript = simulate_call(name, price)
            for line in transcript:
                st.write(line)
            
            st.write("📊 Записываю в CRM...")
            crm_data = analyze_crm(transcript)
            st.json(crm_data)
            
            status.update(label="Завершено!", state="complete")
