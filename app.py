import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, time, timedelta
import io
import os

# --- 1. ГЛОБАЛЬНЫЕ НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(
    page_title="АСКУЭ Аналитика Pro", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# CSS: Убираем лишние отступы и увеличиваем шрифт метрик
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 3rem;}
        div[data-testid="stMetricValue"] {font-size: 22px;}
        h3 {font-size: 20px !important;}
    </style>
""", unsafe_allow_html=True)

# --- 2. ЛОГИКА ПАРСИНГА ФАЙЛОВ ---
@st.cache_data
def parse_askue_files(file_objects, selected_year):
    all_data = []
    
    for file_obj in file_objects:
        try:
            # Декодируем байты в строку
            stringio = io.StringIO(file_obj.getvalue().decode("utf-8", errors='ignore'))
        except Exception:
            continue

        lines = stringio.readlines()
        file_date = None
        
        # Поиск даты в заголовке (формат 30917:MMDD)
        if len(lines) > 0:
            header = lines[0]
            if "30917" in header:
                parts = header.split(":")
                if len(parts) >= 2 and len(parts[1]) == 4 and parts[1].isdigit():
                    try:
                        file_date = datetime(selected_year, int(parts[1][:2]), int(parts[1][2:])).date()
                    except: pass
        
        if not file_date: continue
            
        # Парсинг строк данных
        for line in lines:
            if line.startswith("(") and "):" in line:
                parts = line.split(":")
                full_code = parts[0].replace("(", "").replace(")", "")
                
                if len(full_code) >= 6:
                    main = full_code[:5]
                    suf = full_code[-1]
                    
                    # Фильтр: коды 69347/69339 и каналы 1-4
                    if main in ["69347", "69339"] and suf in ["1", "2", "3", "4"]:
                        type_map = {
                            "1": "Актив Прием (кВт)", "2": "Актив Отдача (кВт)",
                            "3": "Реактив Прием (кВАр)", "4": "Реактив Отдача (кВАр)"
                        }
                        # Данные начинаются со 2-го элемента (индекс 2), 48 получесовок
                        if len(parts) >= 50:
                            for i in range(1, 49):
                                try: val = float(parts[i+1].replace(",", "."))
                                except: val = 0.0
                                
                                ts = datetime.combine(file_date, datetime.min.time()) + timedelta(minutes=i*30)
                                all_data.append({
                                    "DateTime": ts, "Date": file_date, "Time": ts.time(),
                                    "MeterID": main + "0", 
                                    "Type": type_map.get(suf, "?"),
                                    "Suffix": int(suf), 
                                    "Value": val
                                })

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

# --- 3. ЗАГРУЗКА ИЗ ПАПКИ (С ИСПРАВЛЕНИЕМ КЭША) ---
def load_files_from_folder(folder_path):
    collected = []
    try:
        if os.path.isdir(folder_path):
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(".txt"):
                    fpath = os.path.join(folder_path, fname)
                    with open(fpath, "rb") as f:
                        obj = io.BytesIO(f.read())
                        # ВАЖНО: Передаем абсолютный путь для корректного кэширования
                        obj.name = os.path.abspath(fpath)
                        collected.append(obj)
            return collected, None
        return [], "Папка не найдена или путь указан неверно."
    except Exception as e: return [], str(e)

# --- 4. БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("⚙️ Управление")
    selected_year = st.number_input("Год данных", 2000, 2100, datetime.now().year)
    
    st.subheader("📂 Источник данных")
    tab_f1, tab_f2 = st.tabs(["Файлы", "Папка"])
    
    final_files = []
    
    # Вкладка 1: Drag & Drop
    with tab_f1:
        upl = st.file_uploader("Перетащите файлы .txt", accept_multiple_files=True, type="txt")
        if upl: final_files.extend(upl)
        
    # Вкладка 2: Путь к папке
    with tab_f2:
        fp = st.text_input("Путь к папке:", placeholder="C:\\Data\\Askue")
        if fp:
            loc, err = load_files_from_folder(fp)
            if err: st.error(err)
            elif loc: 
                st.success(f"Найдено {len(loc)} шт.")
                final_files.extend(loc)
    
    st.divider()
    
    st.subheader("🎨 Вид графиков")
    chart_h = st.slider("Высота (px)", 300, 1200, 600, 50, help="Растяните график по вертикали для детального просмотра")
    line_w = st.slider("Толщина линий", 1, 5, 2)
    show_pts = st.checkbox("Показывать точки на линии", False)

# --- 5. ОСНОВНОЙ ЭКРАН ---
st.title("⚡ Энергомониторинг Dashboard")

if final_files:
    with st.spinner(f'Обработка {len(final_files)} файлов...'):
        df = parse_askue_files(final_files, selected_year)
    
    if not df.empty:
        # --- БЛОК ФИЛЬТРОВ (Expandable) ---
        with st.expander("🔎 Фильтры данных", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1: 
                meters = sorted(df['MeterID'].unique())
                sel_meters = st.multiselect("Точки учета:", meters, default=meters)
            with c2: 
                types = sorted(df['Type'].unique())
                sel_types = st.multiselect("Параметры:", types, default=["Актив Прием (кВт)"])
            with c3: 
                d_min, d_max = df['Date'].min(), df['Date'].max()
                d_rng = st.date_input("Период:", [d_min, d_max], min_value=d_min, max_value=d_max)

        # Применение фильтров
        if len(d_rng) == 2:
            df_v = df[(df['MeterID'].isin(sel_meters)) & (df['Type'].isin(sel_types)) & (df['Date'] >= d_rng[0]) & (df['Date'] <= d_rng[1])]
        else:
            df_v = df[(df['MeterID'].isin(sel_meters)) & (df['Type'].isin(sel_types))]

        if df_v.empty:
            st.warning("Нет данных для выбранных фильтров.")
        else:
            # --- KPI ПАНЕЛЬ ---
            st.markdown("### 📊 Сводка за период")
            k1, k2, k3, k4 = st.columns(4)
            
            act_sum = df_v[df_v['Type'].str.contains("Актив")]['Value'].sum()
            react_sum = df_v[df_v['Type'].str.contains("Реактив")]['Value'].sum()
            peak = df_v['Value'].max()
            peak_t = df_v.loc[df_v['Value'].idxmax()]['DateTime']

            k1.metric("Актив (Энергия)", f"{act_sum:,.0f} кВт·ч".replace(",", " "))
            k2.metric("Реактив (Энергия)", f"{react_sum:,.0f} кВАр·ч".replace(",", " "))
            k3.metric("Пиковая мощность", f"{peak:,.2f} кВт")
            k4.metric("Время пика", peak_t.strftime('%d.%m %H:%M'))
            st.divider()

            # --- ВКЛАДКИ КОНТЕНТА ---
            t1, t2, t3, t4 = st.tabs(["📈 График нагрузки", "📅 Суточные итоги", "🔥 Тепловая карта", "🧠 Умный анализ"])

            # 1. ДЕТАЛЬНЫЙ ГРАФИК
            with t1:
                fig = go.Figure()
                # Умный заголовок оси Y
                has_kw = any("кВт" in t for t in sel_types)
                has_kvar = any("кВАр" in t for t in sel_types)
                if has_kw and not has_kvar: y_title = "Активная мощность (кВт)"
                elif not has_kw and has_kvar: y_title = "Реактивная мощность (кВАр)"
                else: y_title = "Мощность (кВт) / Реактив (кВАр)"

                for m in sel_meters:
                    for t in sel_types:
                        sub = df_v[(df_v['MeterID'] == m) & (df_v['Type'] == t)]
                        if not sub.empty:
                            fig.add_trace(go.Scatter(
                                x=sub['DateTime'], y=sub['Value'],
                                mode='lines+markers' if show_pts else 'lines',
                                name=f"{m} {t.split('(')[0]}", # Короткое имя в легенде
                                line=dict(width=line_w),
                                hovertemplate='<b>%{y:.2f}</b><br>%{x|%d.%m %H:%M}'
                            ))
                
                fig.update_layout(
                    height=chart_h, # Регулируемая высота
                    template="plotly_white", # Чистый белый стиль
                    legend=dict(orientation="h", y=1.02, x=0), # Легенда сверху
                    margin=dict(l=10, r=10, t=30, b=10), 
                    hovermode="x unified",
                    yaxis=dict(title=y_title, showgrid=True),
                    xaxis=dict(title="Время", showgrid=True, rangeslider=dict(visible=True))
                )
                st.plotly_chart(fig, use_container_width=True)

            # 2. СУТОЧНЫЕ ИТОГИ
            with t2:
                d_g = df_v.groupby(['Date', 'Type', 'MeterID'])['Value'].sum().reset_index()
                fig_b = px.bar(d_g, x='Date', y='Value', color='Type', barmode='group', title="Суточное потребление")
                fig_b.update_layout(
                    height=chart_h * 0.8, 
                    template="plotly_white", 
                    yaxis_title="Энергия (кВт·ч / кВАр·ч)",
                    legend=dict(orientation="h", y=1.02, x=0)
                )
                st.plotly_chart(fig_b, use_container_width=True)

            # 3. ТЕПЛОВАЯ КАРТА (УЛУЧШЕННАЯ)
            with t3:
                c_h1, c_h2, c_h3 = st.columns([1, 1, 1])
                with c_h1: hm_m = st.selectbox("Точка учета:", meters, key="hm_meter")
                with c_h2: hm_t = st.selectbox("Параметр:", types, key="hm_type")
                with c_h3: show_vals = st.checkbox("Показать значения (кВт)", value=False)

                dh = df[(df['MeterID'] == hm_m) & (df['Type'] == hm_t)].copy()
                
                if not dh.empty:
                    # Подготовка матрицы
                    dh['TimeStr'] = dh['Time'].apply(lambda x: x.strftime('%H:%M'))
                    dh['DateStr'] = dh['Date'].apply(lambda x: x.strftime('%d.%m'))
                    
                    pivot_data = dh.pivot_table(index='TimeStr', columns='DateStr', values='Value', aggfunc='sum')
                    
                    # Сортировка времени
                    pivot_data.index = pd.to_datetime(pivot_data.index, format='%H:%M').time
                    pivot_data.sort_index(inplace=True)
                    pivot_data.index = [t.strftime('%H:%M') for t in pivot_data.index]

                    fig_h = px.imshow(
                        pivot_data,
                        labels=dict(x="Дата", y="Время", color="Значение"),
                        x=pivot_data.columns,
                        y=pivot_data.index,
                        aspect="auto",
                        color_continuous_scale='RdYlGn_r',
                        text_auto='.0f' if show_vals else False
                    )

                    fig_h.update_layout(
                        height=chart_h if not show_vals else max(800, chart_h),
                        title=f"Матрица нагрузок: {hm_m} ({hm_t})",
                        xaxis_nticks=30
                    )
                    fig_h.update_xaxes(side="top") # Даты сверху
                    st.plotly_chart(fig_h, use_container_width=True)
                else:
                    st.warning("Нет данных для отображения карты.")

            # 4. УМНЫЙ АНАЛИЗ
            with t4:
                st.subheader("📊 Анализ эффективности энергопотребления")
                
                # Работаем с активной энергией
                df_act = df[(df['MeterID'].isin(sel_meters)) & (df['Suffix'] == 1)] # Актив Прием
                
                if not df_act.empty:
                    c_a1, c_a2 = st.columns(2)
                    
                    # Метрика: Коэффициент заполнения
                    avg_p = df_act['Value'].mean()
                    max_p = df_act['Value'].max()
                    k_zap = avg_p / max_p if max_p > 0 else 0
                    
                    with c_a1:
                        st.markdown(f"""
                        **Коэффициент заполнения графика ($K_{{zap}}$):** `{k_zap:.2f}`
                        """)
                        if k_zap > 0.7: st.success("✅ График ровный (эффективное потребление).")
                        elif k_zap > 0.4: st.info("ℹ️ Средняя неравномерность.")
                        else: st.warning("⚠️ Высокая неравномерность (пиковые нагрузки).")

                    # Метрика: День / Ночь
                    day_start, day_end = time(8,0), time(20,0)
                    mask_day = (df_act['Time'] >= day_start) & (df_act['Time'] < day_end)
                    v_day = df_act[mask_day]['Value'].sum()
                    v_night = df_act[~mask_day]['Value'].sum()
                    
                    with c_a2:
                        fig_pie = px.pie(values=[v_day, v_night], names=['День (08-20)', 'Ночь (20-08)'], hole=0.4, title="Распределение по зонам суток")
                        fig_pie.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie, use_container_width=True)

                st.divider()
                
                # Метрика: Cos Phi и Scatter
                df_calc = df[df['Suffix'].isin([1, 3])].copy()
                if not df_calc.empty:
                    piv = df_calc.pivot_table(index=['DateTime', 'MeterID'], columns='Suffix', values='Value').reset_index()
                    if 1 in piv.columns and 3 in piv.columns:
                        piv['S'] = np.sqrt(piv[1]**2 + piv[3]**2)
                        piv['CosPhi'] = np.where(piv['S'] > 0, piv[1] / piv['S'], 0)
                        
                        st.markdown("#### 📉 Реактивная мощность и Cos φ")
                        
                        fig_cos = px.line(piv, x='DateTime', y='CosPhi', color='MeterID', title="Динамика Cos φ")
                        fig_cos.add_hline(y=0.96, line_dash="dash", line_color="red", annotation_text="Норма 0.96")
                        fig_cos.update_layout(height=400, yaxis_title="Cos φ", template="plotly_white", yaxis_range=[0.5, 1.02])
                        st.plotly_chart(fig_cos, use_container_width=True)
                        
                        st.markdown("**Характер нагрузки (Актив vs Реактив)**")
                        fig_scat = px.scatter(piv, x=1, y=3, color='MeterID', trendline="ols",
                                              labels={"1": "Актив (кВт)", "3": "Реактив (кВАр)"})
                        fig_scat.update_layout(height=500, template="plotly_white")
                        st.plotly_chart(fig_scat, use_container_width=True)
                    else:
                        st.info("Для анализа качества нужны данные по активной и реактивной энергии.")

else:
    st.markdown("""
    <div style='text-align: center; margin-top: 100px; color: #888;'>
        <h1>👋 Добро пожаловать</h1>
        <p>Загрузите файлы в меню слева (Drag & Drop) или укажите папку.</p>
    </div>
    """, unsafe_allow_html=True)