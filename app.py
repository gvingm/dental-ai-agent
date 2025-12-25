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
        
    openai_key = st.text_input("OpenAI Key", value=st.session_state["openai_key"], type="password")
    serper_key = st.text_input("Serper Key", value=st.session_state["serper_key"], type="password")
    
    if st.button("Save & Apply"):
        st.session_state["openai_key"] = openai_key
        st.session_state["serper_key"] = serper_key
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["SERPER_API_KEY"] = serper_key
        st.success("Keys Saved!")

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.session_state.get("openai_key", DEFAULT_OPENAI)
    
if not os.environ.get("SERPER_API_KEY"):
    os.environ["SERPER_API_KEY"] = st.session_state.get("serper_key", DEFAULT_SERPER)

# === ЛОГИКА (ФУНКЦИИ) ===

def get_llm():
    return ChatOpenAI(
        model="openai/gpt-4o",
        base_url="https://openrouter.ai/api/v1",
    	api_key=os.environ["OPENAI_API_KEY"],
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
    
    msg1 = llm.invoke([client_sys, HumanMessage(content=steps[0])])
    text1 = msg1.content.replace("CLIENT:", "").strip()
    transcript.append(f"👤 **КЛИЕНТ:** {text1}")
    history.append(HumanMessage(content=f"CLIENT: {text1}"))
    
    msg2 = llm.invoke(history)
    text2 = msg2.content.replace("ADMIN:", "").strip()
    transcript.append(f"👩‍⚕️ **АДМИН:** {text2}")
    history.append(SystemMessage(content=f"ADMIN: {text2}"))
    
    msg3 = llm.invoke([client_sys, HumanMessage(content=steps[1].format(prev=text2))])
    text3 = msg3.content.replace("CLIENT:", "").strip()
    transcript.append(f"👤 **КЛИЕНТ:** {text3}")
    history.append(HumanMessage(content=f"CLIENT: {text3}"))
    
    msg4 = llm.invoke(history)
    text4 = msg4.content.replace("ADMIN:", "").strip()
    transcript.append(f"👩‍⚕️ **АДМИН:** {text4}")
    history.append(SystemMessage(content=f"ADMIN: {text4}"))
    
    msg5 = llm.invoke([client_sys, HumanMessage(content=steps[2])])
    text5 = msg5.content.replace("CLIENT:", "").strip()
    transcript.append(f"👤 **КЛИЕНТ:** {text5}")
    
    msg6 = llm.invoke(history + [HumanMessage(content=f"CLIENT: {text5}")])
    text6 = msg6.content.replace("ADMIN:", "").strip()
    transcript.append(f"👩‍⚕️ **АДМИН:** {text6}")

    return transcript

def analyze_crm(transcript_list):
    llm = get_llm()
    text = "\n".join(transcript_list)
    crm_template = """
    Проанализируй диалог. Верни JSON со следующими полями на русском языке:
    {{
      "статус": "...",  # Например, "запрос", "забронировано", "отказ"
      "цена_упомянута": "...", # Например, "10000 руб" или "не указана"
      "результат_звонка": "..." # Краткое описание результата
    }}
    Текст: {t}
    """
    chain = ChatPromptTemplate.from_template(crm_template) | llm | StrOutputParser()
    raw_response = chain.invoke({"t": text})

    try:
        clean_json_str = raw_response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json_str)
    except json.JSONDecodeError:
        return {"error": "Не удалось разобрать JSON от LLM", "raw_response": raw_response}


def search_cheapest_clinic(query):
    search = GoogleSerperAPIWrapper()
    try:
        raw_results = search.results(query)
        organic = raw_results.get("organic", [])
    except Exception as e:
        return None, None, f"Ошибка поиска Google: {str(e)}"
    
    if not organic:
        return None, None, "В Google ничего не найдено"

    text_data = ""
    for item in organic:
        text_data += f"Клиника: {item.get('title')}\nОписание: {item.get('snippet')}\n\n"
        
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
        res = chain.invoke({"text": text_data}).strip()
        
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
        return None, None, f"Ошибка парсинга LLM: {str(e)}"

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