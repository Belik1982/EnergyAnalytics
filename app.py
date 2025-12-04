import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, time, timedelta
import io
import os # <--- Добавили библиотеку для работы с файловой системой

# --- 1. ГЛОБАЛЬНЫЕ НАСТРОЙКИ ИНТЕРФЕЙСА ---
st.set_page_config(
    page_title="АСКУЭ Аналитика Pro", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        div[data-testid="stMetricValue"] {font-size: 24px;}
    </style>
""", unsafe_allow_html=True)

# --- 2. ЛОГИКА ПАРСИНГА ---
@st.cache_data
def parse_askue_files(file_objects, selected_year):
    # file_objects - это список BytesIO объектов (не важно, загружены они или считаны с диска)
    all_data = []
    
    for file_obj in file_objects:
        # Декодируем байты в строку
        try:
            stringio = io.StringIO(file_obj.getvalue().decode("utf-8", errors='ignore'))
        except Exception:
            continue # Если файл битый

        lines = stringio.readlines()
        file_date = None
        
        # Поиск даты
        if len(lines) > 0:
            header = lines[0]
            if "30917" in header:
                parts = header.split(":")
                if len(parts) >= 2:
                    date_code = parts[1]
                    if len(date_code) == 4 and date_code.isdigit():
                        try:
                            file_date = datetime(selected_year, int(date_code[:2]), int(date_code[2:])).date()
                        except: pass
        
        if not file_date: continue
            
        # Парсинг строк
        for line in lines:
            if line.startswith("(") and "):" in line:
                parts = line.split(":")
                full_code_raw = parts[0].replace("(", "").replace(")", "")
                
                if len(full_code_raw) >= 6:
                    main_code = full_code_raw[:5]
                    suffix = full_code_raw[-1]
                    
                    if main_code in ["69347", "69339"] and suffix in ["1", "2", "3", "4"]:
                        type_map = {
                            "1": "Актив Прием (kW)", "2": "Актив Отдача (kW)",
                            "3": "Реактив Прием (kVar)", "4": "Реактив Отдача (kVar)"
                        }
                        
                        if len(parts) >= 50:
                            for i in range(1, 49):
                                try:
                                    val = float(parts[i+1].replace(",", "."))
                                except: val = 0.0
                                
                                timestamp = datetime.combine(file_date, datetime.min.time()) + timedelta(minutes=i*30)
                                
                                all_data.append({
                                    "DateTime": timestamp,
                                    "Date": file_date,
                                    "Time": timestamp.time(),
                                    "MeterID": main_code + "0",
                                    "Type": type_map.get(suffix, "Unknown"),
                                    "Suffix": int(suffix),
                                    "Value": val
                                })

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

# --- 3. ФУНКЦИЯ ЧТЕНИЯ ИЗ ПАПКИ (ИСПРАВЛЕННАЯ) ---
def load_files_from_folder(folder_path):
    collected_files = []
    try:
        # Проверяем, существует ли папка
        if os.path.isdir(folder_path):
            # Перебираем файлы
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(".txt"):
                    filepath = os.path.join(folder_path, filename)
                    # Читаем в бинарном режиме
                    with open(filepath, "rb") as f:
                        content = f.read()
                        # Создаем объект BytesIO
                        bytes_obj = io.BytesIO(content)
                        # !!! ИСПРАВЛЕНИЕ: Передаем полный абсолютный путь, 
                        # чтобы Streamlit мог корректно проверить файл для кэша
                        bytes_obj.name = os.path.abspath(filepath) 
                        collected_files.append(bytes_obj)
            return collected_files, None
        else:
            return [], "Папка не найдена. Проверьте путь."
    except Exception as e:
        return [], str(e)

# --- 4. БОКОВАЯ ПАНЕЛЬ (НАСТРОЙКИ) ---
with st.sidebar:
    st.header("⚙️ Панель управления")
    selected_year = st.number_input("Год данных", 2000, 2100, datetime.now().year)
    
    # === ВАРИАНТ 1: ЗАГРУЗКА ФАЙЛОВ ===
    st.subheader("📂 Загрузка данных")
    
    # Вкладки для методов загрузки
    load_tab1, load_tab2 = st.tabs(["Файлы", "Папка"])
    
    final_file_list = [] # Сюда соберем файлы из обоих источников
    
    with load_tab1:
        uploaded_files = st.file_uploader("Перетащите файлы сюда", accept_multiple_files=True, type="txt")
        if uploaded_files:
            final_file_list.extend(uploaded_files)
            
    with load_tab2:
        folder_path = st.text_input("Путь к папке:", placeholder=r"C:\Данные\АСКУЭ")
        st.caption("Скопируйте путь из адресной строки проводника и нажмите Enter.")
        
        # Кнопка для явной загрузки (хотя Enter тоже сработает)
        if folder_path:
            local_files, error_msg = load_files_from_folder(folder_path)
            if error_msg:
                st.error(error_msg)
            elif local_files:
                st.success(f"Найдено {len(local_files)} файлов .txt")
                final_file_list.extend(local_files)
            else:
                st.warning("В папке нет файлов .txt")
    
    st.divider()
    
    # Блок внешнего вида
    st.subheader("🎨 Вид")
    chart_height = st.slider("Высота графика (px)", 300, 1200, 600, 50)
    line_width = st.slider("Толщина линий", 1, 5, 2)
    show_markers = st.checkbox("Показывать точки", value=False)
    
    st.divider()

# --- 5. ОСНОВНОЙ ЭКРАН ---
st.title("⚡ Энергомониторинг Dashboard")

if final_file_list:
    # Передаем combined список (и загруженные вручную, и из папки)
    with st.spinner(f'Обработка {len(final_file_list)} файлов...'):
        df = parse_askue_files(final_file_list, selected_year)
    
    if not df.empty:
        # --- ФИЛЬТРЫ ---
        with st.expander("🔎 Фильтрация данных", expanded=True):
            col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
            with col_f1:
                all_meters = sorted(df['MeterID'].unique())
                sel_meters = st.multiselect("Точки учета:", all_meters, default=all_meters)
            with col_f2:
                all_types = sorted(df['Type'].unique())
                sel_types = st.multiselect("Параметры:", all_types, default=["Актив Прием (kW)"])
            with col_f3:
                min_d, max_d = df['Date'].min(), df['Date'].max()
                date_range = st.date_input("Период:", [min_d, max_d], min_value=min_d, max_value=max_d)

        # Применение фильтров
        if len(date_range) == 2:
            mask = (df['MeterID'].isin(sel_meters)) & (df['Type'].isin(sel_types)) & \
                   (df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])
            df_view = df[mask]
        else:
            df_view = df[df['MeterID'].isin(sel_meters) & df['Type'].isin(sel_types)]

        if df_view.empty:
            st.warning("Нет данных для выбранных фильтров.")
        else:
            # --- KPI ---
            st.markdown("### 📊 Ключевые показатели")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            total_active = df_view[df_view['Type'].str.contains("Актив")]['Value'].sum()
            max_peak = df_view['Value'].max()
            peak_time = df_view.loc[df_view['Value'].idxmax()]['DateTime']
            
            kpi1.metric("Всего (Актив)", f"{total_active:,.0f} кВт·ч".replace(",", " "))
            kpi2.metric("Пик нагрузки", f"{max_peak:,.2f} кВт")
            kpi3.metric("Время пика", peak_time.strftime('%d.%m %H:%M'))
            kpi4.metric("Источников данных", f"{len(final_file_list)}") # Показываем кол-во файлов
            st.divider()

            # --- ВКЛАДКИ ---
            tab_main, tab_daily, tab_heat, tab_anal = st.tabs(["📈 Детальный график", "📅 Суточные", "🔥 Тепловая карта", "🧠 Анализ"])

            # 1. ГРАФИК
            with tab_main:
                fig = go.Figure()
                for m_id in sel_meters:
                    for t_type in sel_types:
                        subset = df_view[(df_view['MeterID'] == m_id) & (df_view['Type'] == t_type)]
                        if not subset.empty:
                            fig.add_trace(go.Scatter(
                                x=subset['DateTime'], y=subset['Value'],
                                mode='lines+markers' if show_markers else 'lines',
                                name=f"{m_id} - {t_type}",
                                line=dict(width=line_width),
                                hovertemplate='%{y:.2f} <br>%{x|%d.%m %H:%M}'
                            ))
                fig.update_layout(
                    height=chart_height, template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified",
                    xaxis=dict(rangeslider=dict(visible=True), showgrid=True),
                    yaxis=dict(showgrid=True, title="Мощность")
                )
                st.plotly_chart(fig, use_container_width=True)

            # 2. СУТОЧНЫЕ
            with tab_daily:
                daily_grp = df_view.groupby(['Date', 'Type', 'MeterID'])['Value'].sum().reset_index()
                fig_bar = px.bar(daily_grp, x='Date', y='Value', color='Type', barmode='group', title="Потребление по дням")
                fig_bar.update_layout(height=chart_height * 0.8, template="plotly_white")
                st.plotly_chart(fig_bar, use_container_width=True)

            # 3. HEATMAP
            with tab_heat:
                col_h1, col_h2 = st.columns(2)
                with col_h1: hm_meter = st.selectbox("Точка:", all_meters)
                with col_h2: hm_type = st.selectbox("Параметр:", all_types)
                df_heat = df[(df['MeterID'] == hm_meter) & (df['Type'] == hm_type)].copy()
                if not df_heat.empty:
                    df_heat['TimeStr'] = df_heat['Time'].astype(str)
                    df_heat['DateStr'] = df_heat['Date'].astype(str)
                    fig_heat = px.density_heatmap(df_heat, x='DateStr', y='TimeStr', z='Value', nbinsy=48, color_continuous_scale='RdYlGn_r')
                    fig_heat.update_layout(height=chart_height, yaxis=dict(autorange="reversed"), title=f"Карта: {hm_meter}")
                    st.plotly_chart(fig_heat, use_container_width=True)

            # 4. АНАЛИЗ
            with tab_anal:
                df_calc = df[df['Suffix'].isin([1, 3])].copy()
                if not df_calc.empty:
                    pivoted = df_calc.pivot_table(index=['DateTime', 'MeterID'], columns='Suffix', values='Value').reset_index()
                    if 1 in pivoted.columns and 3 in pivoted.columns:
                        pivoted['S'] = np.sqrt(pivoted[1]**2 + pivoted[3]**2)
                        pivoted['CosPhi'] = np.where(pivoted['S'] > 0, pivoted[1] / pivoted['S'], 0)
                        fig_cos = px.line(pivoted, x='DateTime', y='CosPhi', color='MeterID', title="Cos φ")
                        fig_cos.add_hline(y=0.96, line_dash="dot", line_color="red")
                        fig_cos.update_layout(height=chart_height * 0.8, yaxis_range=[0.6, 1.02], template="plotly_white")
                        st.plotly_chart(fig_cos, use_container_width=True)
                    else: st.warning("Нет данных Актив+Реактив.")

    else:
        st.error("В загруженных файлах не найдены нужные коды.")

else:
    # LANDING PAGE
    st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <h1>⚡ Энергомониторинг Dashboard</h1>
        <p style='color: gray;'>
            Укажите путь к папке или загрузите файлы вручную в меню слева.
        </p>
    </div>
    """, unsafe_allow_html=True)