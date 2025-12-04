import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, time, timedelta
import io
import os
import google.generativeai as genai

# --- 1. ГЛОБАЛЬНЫЕ НАСТРОЙКИ ---
st.set_page_config(
    page_title="АСКУЭ Pro", 
    layout="wide", 
    page_icon="🏭",
    initial_sidebar_state="expanded"
)

# --- 2. УПРАВЛЕНИЕ СТИЛЯМИ (CSS) ---
def apply_custom_css(font_scale):
    """
    Применяет CSS стили с учетом масштаба шрифта.
    font_scale: 1.0 (обычный) или 1.25 (крупный)
    """
    base_size = 16 * font_scale
    metric_size = 24 * font_scale
    
    st.markdown(f"""
        <style>
            /* Глобальный размер шрифта */
            html, body, [class*="css"] {{
                font-size: {base_size}px;
            }}
            /* Метрики (KPI) */
            div[data-testid="stMetricValue"] {{
                font-size: {metric_size}px !important;
                color: #0068c9;
                font-weight: 600;
            }}
            /* Теги мультиселекта (фильтры) */
            span[data-baseweb="tag"] {{
                font-size: {14 * font_scale}px;
            }}
            /* Заголовки */
            h1 {{ font-size: {32 * font_scale}px !important; }}
            h2 {{ font-size: {26 * font_scale}px !important; }}
            h3 {{ font-size: {22 * font_scale}px !important; }}
            
            /* ИСПРАВЛЕНИЕ: Увеличенный отступ сверху, чтобы не обрезался заголовок */
            .block-container {{
                padding-top: 4rem; 
                padding-bottom: 3rem;
            }}
        </style>
    """, unsafe_allow_html=True)

# --- 3. ФОРМАТИРОВАНИЕ ЧИСЕЛ ---
def fmt_num(val):
    """Форматирует числа: 2817.5 -> '2 817.5' (пробел как разделитель)"""
    if pd.isna(val): return "-"
    if val > 100:
        s = "{:,.0f}".format(val) # Целые для больших чисел
    else:
        s = "{:,.2f}".format(val) # Дробные для малых
    return s.replace(",", " ")

# --- 4. ПАРСИНГ ФАЙЛОВ ---
@st.cache_data
def parse_askue_files(file_objects, selected_year):
    all_data = []
    
    for file_obj in file_objects:
        try:
            content = file_obj.getvalue().decode("utf-8", errors='ignore')
            stringio = io.StringIO(content)
        except Exception:
            continue

        lines = stringio.readlines()
        file_date = None
        
        # Поиск даты в заголовке
        if len(lines) > 0:
            header = lines[0]
            if "30917" in header:
                parts = header.split(":")
                if len(parts) >= 2 and len(parts[1]) == 4 and parts[1].isdigit():
                    try:
                        file_date = datetime(selected_year, int(parts[1][:2]), int(parts[1][2:])).date()
                    except: pass
        
        if not file_date: continue
            
        for line in lines:
            if line.startswith("(") and "):" in line:
                parts = line.split(":")
                full_code = parts[0].replace("(", "").replace(")", "")
                
                if len(full_code) >= 6:
                    main = full_code[:5]
                    try: suf = int(full_code[-1])
                    except: suf = 0
                    
                    # Фильтр счетчиков и каналов
                    if main in ["69347", "69339"] and suf in [1, 2, 3, 4]:
                        type_label = "?"
                        unit = ""
                        # Интерпретация каналов
                        if suf == 2: 
                            type_label = "Акт. Потр."
                            unit = "кВт"
                        elif suf == 4: 
                            type_label = "Реакт. Потр."
                            unit = "кВАр"
                        elif suf == 1: 
                            type_label = "Акт. Переток"
                            unit = "кВт"
                        elif suf == 3: 
                            type_label = "Реакт. Переток"
                            unit = "кВАр"

                        if len(parts) >= 50:
                            for i in range(1, 49):
                                try: val = float(parts[i+1].replace(",", "."))
                                except: val = 0.0
                                
                                ts = datetime.combine(file_date, datetime.min.time()) + timedelta(minutes=i*30)
                                all_data.append({
                                    "DateTime": ts, "Date": file_date, "Time": ts.time(),
                                    "MeterID": main, 
                                    "Type": f"{type_label} ({unit})", 
                                    "ShortType": type_label,
                                    "Unit": unit,
                                    "Suffix": suf, 
                                    "Value": val
                                })

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

# --- 5. ФУНКЦИИ ИИ (GEMINI) ---
def get_ai_response(api_key, model_name, messages):
    try:
        genai.configure(api_key=api_key)
        gemini_history = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
            
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(gemini_history)
        return response.text
    except Exception as e:
        return f"⚠️ Ошибка API ({model_name}): {str(e)}"

# --- 6. ЗАГРУЗКА ФАЙЛОВ ИЗ ПАПКИ ---
def load_files_from_folder(folder_path):
    collected = []
    try:
        if os.path.isdir(folder_path):
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(".txt"):
                    fpath = os.path.join(folder_path, fname)
                    with open(fpath, "rb") as f:
                        obj = io.BytesIO(f.read())
                        obj.name = os.path.abspath(fpath)
                        collected.append(obj)
            return collected, None
        return [], "Папка не найдена."
    except Exception as e: return [], str(e)

# --- 7. БОКОВАЯ ПАНЕЛЬ (НАСТРОЙКИ + ЧАТ) ---
with st.sidebar:
    st.title("🎛️ Панель управления")
    
    # 1. ВНЕШНИЙ ВИД
    with st.expander("👁️ Вид и Шрифт", expanded=False):
        font_mode = st.radio("Размер текста", ["Нормальный", "Крупный (для чтения)"])
        # Применяем стили сразу
        font_scale = 1.25 if font_mode == "Крупный (для чтения)" else 1.0
        apply_custom_css(font_scale)
        
        chart_h = st.slider("Высота графика", 300, 1000, 500, 50)
        line_w = st.slider("Толщина линий", 1, 4, 2)
        show_pts = st.checkbox("Точки на графике", value=False)
    
    # 2. ЗАГРУЗКА ДАННЫХ
    with st.expander("📂 Загрузка данных", expanded=True):
        selected_year = st.number_input("Год", 2000, 2100, datetime.now().year)
        
        tab_f1, tab_f2 = st.tabs(["Файлы", "Папка"])
        final_files = []
        with tab_f1:
            upl = st.file_uploader("Файлы .txt", accept_multiple_files=True)
            if upl: final_files.extend(upl)
        with tab_f2:
            fp = st.text_input("Путь:", placeholder="C:\\Data")
            if fp:
                loc, err = load_files_from_folder(fp)
                if loc: final_files.extend(loc)
    
    # 3. НАСТРОЙКИ ИИ
    with st.expander("🤖 Настройки ИИ", expanded=False):
        api_key_input = st.text_input("API Key", value="AIzaSyAQu0wQTLYAIIU5hpgQN0BUFuQrvApeUpk", type="password")
        model_options = [
            "gemini-3-pro-preview",
            "gemini-2.5-pro",
            "gemini-2.0-flash", 
            "gemini-1.5-pro", 
            "gemini-1.5-flash"
        ]
        model_name_input = st.selectbox("Модель", model_options, index=0)

    st.divider()

    # 4. ЧАТ С ИИ (ВСЕГДА ВИДЕН)
    st.header("💬 Чат с помощником")
    chat_container = st.container()

# --- 8. ОСНОВНОЙ ЭКРАН ---
st.title("⚡ АСКУЭ Аналитика")

if final_files:
    df = parse_askue_files(final_files, selected_year)
    
    if not df.empty:
        # --- ФИЛЬТРЫ ---
        with st.expander("🔎 Фильтры данных", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1: 
                meters = sorted(df['MeterID'].unique())
                sel_meters = st.multiselect("Счетчики", meters, default=meters)
            with c2: 
                types = sorted(df['Type'].unique())
                # По умолчанию ищем Актив Потребление
                def_t = [t for t in types if "Акт. Потр." in t] 
                if not def_t: def_t = types
                sel_types = st.multiselect("Каналы", types, default=def_t)
            with c3: 
                d_min, d_max = df['Date'].min(), df['Date'].max()
                d_rng = st.date_input("Период", [d_min, d_max])

        # Подготовка данных с учетом фильтров
        if len(d_rng) == 2:
            df_v = df[(df['MeterID'].isin(sel_meters)) & (df['Type'].isin(sel_types)) & (df['Date'] >= d_rng[0]) & (df['Date'] <= d_rng[1])]
            # Для KPI берем каналы 2 и 4 тех же счетчиков за тот же период
            df_kpi = df[(df['MeterID'].isin(sel_meters)) & (df['Date'] >= d_rng[0]) & (df['Date'] <= d_rng[1])]
        else:
            df_v = df[(df['MeterID'].isin(sel_meters)) & (df['Type'].isin(sel_types))]
            df_kpi = df[df['MeterID'].isin(sel_meters)]

        if df_v.empty:
            st.warning("Нет данных для отображения.")
        else:
            # --- РАСЧЕТ KPI ---
            act_val = df_kpi[df_kpi['Suffix'] == 2]['Value'].sum()
            react_val = df_kpi[df_kpi['Suffix'] == 4]['Value'].sum()
            peak = df_v['Value'].max()
            peak_t = df_v.loc[df_v['Value'].idxmax()]['DateTime'].strftime('%d.%m %H:%M') if peak > 0 else "-"
            
            avg_cos = 0
            if act_val > 0:
                avg_cos = act_val / np.sqrt(act_val**2 + react_val**2)

            # --- ОТОБРАЖЕНИЕ KPI ---
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Потребление", f"{fmt_num(act_val)} кВт", help="Активная энергия")
            k2.metric("Реактив", f"{fmt_num(react_val)} кВАр", help="Реактивная энергия")
            k3.metric("Cos φ", f"{avg_cos:.3f}", delta=f"{avg_cos-0.96:.3f}", delta_color="normal")
            k4.metric("Пик", f"{fmt_num(peak)} кВт", delta=f"в {peak_t}", delta_color="off")
            
            # --- ЛОГИКА ЧАТА (В САЙДБАРЕ) ---
            with chat_container:
                # Инициализация истории, если её нет
                if "messages" not in st.session_state:
                    context_prompt = f"""
                    Ты профессиональный энергоаудитор. 
                    ТЕКУЩИЙ КОНТЕКСТ ДАННЫХ:
                    - Период: {d_rng}
                    - Выбранные счетчики: {sel_meters}
                    - Потребление Актив: {act_val:,.0f} кВт
                    - Потребление Реактив: {react_val:,.0f} кВАр
                    - Средний Cos Phi: {avg_cos:.3f} (Норма > 0.96)
                    - Максимальный пик: {peak:.2f} кВт (в {peak_t})
                    
                    Отвечай на русском языке, кратко и понятно. Если спрашивают про график P vs Q, объясни суть реактивной мощности.
                    """
                    st.session_state.messages = [
                        {"role": "user", "content": context_prompt}, 
                        {"role": "model", "content": "Я вижу данные вашей фабрики. Чем могу помочь?"}
                    ]
                
                # Вывод видимых сообщений
                for msg in st.session_state.messages[2:]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # Поле ввода
                if prompt := st.chat_input("Вопрос по данным...", key="sidebar_chat"):
                    if not api_key_input:
                        st.error("Нет API ключа!")
                    else:
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        with st.chat_message("assistant"):
                            with st.spinner("..."):
                                response_text = get_ai_response(api_key_input, model_name_input, st.session_state.messages)
                                st.markdown(response_text)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Кнопка очистки, которая сбрасывает контекст под текущие данные
                if st.button("🧹 Обновить контекст"):
                    del st.session_state.messages
                    st.rerun()

            # --- ГЛАВНЫЕ ВКЛАДКИ ---
            t1, t2, t3, t4 = st.tabs(["📈 Нагрузка", "📅 Итоги", "🔥 Матрица", "🎯 Характер нагрузки (P vs Q)"])

            # 1. ГРАФИК НАГРУЗКИ
            with t1:
                fig = go.Figure()
                y_units = set()
                for m in sel_meters:
                    for t in sel_types:
                        sub = df_v[(df_v['MeterID'] == m) & (df_v['Type'] == t)]
                        if not sub.empty:
                            unit = sub['Unit'].iloc[0] if 'Unit' in sub.columns else ""
                            y_units.add(unit)
                            mode_val = 'lines+markers' if show_pts else 'lines'
                            fig.add_trace(go.Scatter(
                                x=sub['DateTime'], y=sub['Value'],
                                mode=mode_val,
                                name=f"{m} {t}", 
                                line=dict(width=line_w)
                            ))
                
                y_title = " / ".join(list(y_units)) if y_units else "Значение"
                fig.update_layout(
                    height=chart_h, 
                    template="plotly_white", 
                    hovermode="x unified", 
                    legend=dict(orientation="h", y=1.02),
                    yaxis_title=f"Мощность ({y_title})",
                    xaxis_title="Время"
                )
                st.plotly_chart(fig, use_container_width=True)

            # 2. СУТОЧНЫЕ ИТОГИ
            with t2:
                d_g = df_v.groupby(['Date', 'Type'])['Value'].sum().reset_index()
                fig_b = px.bar(d_g, x='Date', y='Value', color='Type', barmode='group')
                fig_b.update_layout(
                    height=chart_h, 
                    template="plotly_white",
                    yaxis_title="Энергия (кВт*ч / кВАр*ч)"
                )
                st.plotly_chart(fig_b, use_container_width=True)

            # 3. ТЕПЛОВАЯ КАРТА
            with t3:
                hm_cols = st.columns([1, 1, 2])
                with hm_cols[0]: 
                    show_vals = st.checkbox("Показать цифры", value=False)

                hm_m = sel_meters[0] if sel_meters else None
                hm_t = next((t for t in sel_types if "Акт. Потр." in t), sel_types[0] if sel_types else None)
                
                if hm_m and hm_t:
                    dh = df[(df['MeterID'] == hm_m) & (df['Type'] == hm_t)].copy()
                    if len(d_rng) == 2: dh = dh[(dh['Date'] >= d_rng[0]) & (dh['Date'] <= d_rng[1])]
                    
                    if not dh.empty:
                        dh['TimeStr'] = dh['Time'].apply(lambda x: x.strftime('%H:%M'))
                        dh['DateStr'] = dh['Date'].apply(lambda x: x.strftime('%d.%m'))
                        p_hm = dh.pivot_table(index='TimeStr', columns='DateStr', values='Value', aggfunc='sum')
                        p_hm.index = pd.to_datetime(p_hm.index, format='%H:%M').time
                        p_hm.sort_index(inplace=True)
                        p_hm.index = [t.strftime('%H:%M') for t in p_hm.index]

                        fig_h = px.imshow(
                            p_hm, aspect="auto", color_continuous_scale='RdYlGn_r', 
                            title=f"Тепловая карта: {hm_m} ({hm_t})",
                            text_auto='.0f' if show_vals else False 
                        )
                        fig_h.update_layout(height=max(600, chart_h))
                        st.plotly_chart(fig_h, use_container_width=True)

            # 4. ХАРАКТЕР НАГРУЗКИ
            with t4:
                st.subheader("Диаграмма рассеяния: Активная (P) vs Реактивная (Q) мощность")
                
                df_c = df[df['MeterID'].isin(sel_meters) & (df['Suffix'].isin([2, 4]))].copy()
                if len(d_rng) == 2: df_c = df_c[(df_c['Date'] >= d_rng[0]) & (df_c['Date'] <= d_rng[1])]
                
                if not df_c.empty:
                    piv = df_c.pivot_table(index=['DateTime', 'MeterID'], columns='Suffix', values='Value').reset_index()
                    
                    if 2 in piv.columns and 4 in piv.columns:
                        # Контрастные цвета
                        fig_s = px.scatter(
                            piv, x=2, y=4, color='MeterID', opacity=0.7,
                            labels={'2': 'Актив P (кВт)', '4': 'Реактив Q (кВАр)'},
                            color_discrete_sequence=["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD"]
                        )
                        
                        # Тренд
                        try:
                            x = piv[2].fillna(0); y = piv[4].fillna(0)
                            if len(x)>1: 
                                k = np.sum(x*y)/np.sum(x**2)
                                x_r = np.linspace(x.min(), x.max(), 10)
                                fig_s.add_trace(go.Scatter(x=x_r, y=k*x_r, mode='lines', 
                                                           line=dict(color='black', dash='dash', width=2), 
                                                           name='Ваш тренд'))
                        except: pass
                        
                        # Идеал
                        max_x = piv[2].max()
                        fig_s.add_trace(go.Scatter(x=[0, max_x], y=[0, max_x*0.29], mode='lines', 
                                                   line=dict(color='green', width=3), 
                                                   name='Идеал (Cos φ 0.96)'))

                        fig_s.update_layout(
                            height=600, 
                            template="plotly_white", 
                            xaxis_title="Активная мощность (кВт)",
                            yaxis_title="Реактивная мощность (кВАр)",
                            legend=dict(orientation="h", y=1.02)
                        )
                        st.plotly_chart(fig_s, use_container_width=True)
                        
                        # ИНСТРУКЦИЯ
                        st.info("""
                        ### 📖 Как понимать этот график?
                        Каждая точка — это 30-минутный интервал работы фабрики.
                        
                        *   **Горизонталь (X)**: Полезная нагрузка. Чем правее, тем больше вы потребляли.
                        *   **Вертикаль (Y)**: Паразитная нагрузка. Чем выше, тем хуже.
                        *   **Зеленая линия**: Граница штрафов (Cos φ = 0.96).
                            *   Точки **под** зеленой линией: Всё отлично.
                            *   Точки **над** зеленой линией: Плохо, идет переплата за реактив.
                        
                        Если много точек красного/синего цвета выше зеленой линии — требуется компенсация (КРМ).
                        """)
                    else:
                        st.warning("Для графика нужны данные по каналам 2 (Актив) и 4 (Реактив).")

else:
    st.info("👈 Пожалуйста, загрузите файлы данных в меню слева.")