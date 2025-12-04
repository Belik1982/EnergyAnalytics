import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, time, timedelta
import io
import os

# --- 1. ГЛОБАЛЬНЫЕ НАСТРОЙКИ ---
st.set_page_config(
    page_title="АСКУЭ Аналитика Pro", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 3rem;}
        div[data-testid="stMetricValue"] {font-size: 22px;}
        h3 {font-size: 20px !important;}
    </style>
""", unsafe_allow_html=True)

# --- 2. ПАРСИНГ ---
@st.cache_data
def parse_askue_files(file_objects, selected_year):
    all_data = []
    
    for file_obj in file_objects:
        try:
            stringio = io.StringIO(file_obj.getvalue().decode("utf-8", errors='ignore'))
        except: continue

        lines = stringio.readlines()
        file_date = None
        
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
                    suf = full_code[-1]
                    
                    if main in ["69347", "69339"] and suf in ["1", "2", "3", "4"]:
                        type_map = {
                            "1": "Актив Прием (кВт)", "2": "Актив Отдача (кВт)",
                            "3": "Реактив Прием (кВАр)", "4": "Реактив Отдача (кВАр)"
                        }
                        if len(parts) >= 50:
                            for i in range(1, 49):
                                try: val = float(parts[i+1].replace(",", "."))
                                except: val = 0.0
                                
                                ts = datetime.combine(file_date, datetime.min.time()) + timedelta(minutes=i*30)
                                all_data.append({
                                    "DateTime": ts, "Date": file_date, "Time": ts.time(),
                                    "MeterID": main + "0", "Type": type_map.get(suf, "?"),
                                    "Suffix": int(suf), "Value": val
                                })

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

# --- 3. ЗАГРУЗКА ИЗ ПАПКИ ---
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

# --- 4. ИНТЕРФЕЙС ---
with st.sidebar:
    st.header("⚙️ Управление")
    selected_year = st.number_input("Год данных", 2000, 2100, datetime.now().year)
    
    tab_f1, tab_f2 = st.tabs(["Файлы", "Папка"])
    final_files = []
    with tab_f1:
        upl = st.file_uploader("Загрузка файлов", accept_multiple_files=True, type="txt")
        if upl: final_files.extend(upl)
    with tab_f2:
        fp = st.text_input("Путь к папке:")
        if fp:
            loc, err = load_files_from_folder(fp)
            if err: st.error(err)
            elif loc: 
                st.success(f"Найдено {len(loc)} шт.")
                final_files.extend(loc)
    
    st.divider()
    st.subheader("🎨 Настройки графика")
    chart_h = st.slider("Высота", 300, 1200, 600, 50)
    line_w = st.slider("Толщина линии", 1, 5, 2)
    show_pts = st.checkbox("Показывать точки", False)

# --- ОСНОВНОЙ КОД ---
st.title("⚡ Энергомониторинг Dashboard")

if final_files:
    with st.spinner(f'Обработка...'):
        df = parse_askue_files(final_files, selected_year)
    
    if not df.empty:
        # Фильтры
        with st.expander("🔎 Фильтры данных", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1: meters = st.multiselect("Точки учета:", sorted(df['MeterID'].unique()), default=sorted(df['MeterID'].unique()))
            with c2: types = st.multiselect("Параметры:", sorted(df['Type'].unique()), default=["Актив Прием (кВт)"])
            with c3: 
                d_min, d_max = df['Date'].min(), df['Date'].max()
                d_rng = st.date_input("Период:", [d_min, d_max], min_value=d_min, max_value=d_max)

        # Применение фильтров
        if len(d_rng) == 2:
            df_v = df[(df['MeterID'].isin(meters)) & (df['Type'].isin(types)) & (df['Date'] >= d_rng[0]) & (df['Date'] <= d_rng[1])]
        else:
            df_v = df[(df['MeterID'].isin(meters)) & (df['Type'].isin(types))]

        if df_v.empty:
            st.warning("Нет данных.")
        else:
            # --- KPI ---
            st.markdown("### 📊 Обзор за период")
            k1, k2, k3, k4 = st.columns(4)
            
            act_sum = df_v[df_v['Type'].str.contains("Актив")]['Value'].sum()
            react_sum = df_v[df_v['Type'].str.contains("Реактив")]['Value'].sum()
            peak = df_v['Value'].max()
            peak_t = df_v.loc[df_v['Value'].idxmax()]['DateTime']

            k1.metric("Актив (Энергия)", f"{act_sum:,.0f} кВт·ч".replace(",", " "))
            k2.metric("Реактив (Энергия)", f"{react_sum:,.0f} кВАр·ч".replace(",", " "))
            k3.metric("Макс. Мощность", f"{peak:,.2f} кВт")
            k4.metric("Время пика", peak_t.strftime('%d.%m %H:%M'))
            st.divider()

            t1, t2, t3, t4 = st.tabs(["📈 График", "📅 Сутки", "🔥 Карта", "🧠 Анализ"])

            # 1. ГРАФИК
            with t1:
                fig = go.Figure()
                # Определяем подпись оси Y
                has_kw = any("кВт" in t for t in types)
                has_kvar = any("кВАр" in t for t in types)
                if has_kw and not has_kvar: y_title = "Активная мощность (кВт)"
                elif not has_kw and has_kvar: y_title = "Реактивная мощность (кВАр)"
                else: y_title = "Мощность (кВт) / Реактив (кВАр)"

                for m in meters:
                    for t in types:
                        sub = df_v[(df_v['MeterID'] == m) & (df_v['Type'] == t)]
                        if not sub.empty:
                            fig.add_trace(go.Scatter(
                                x=sub['DateTime'], y=sub['Value'],
                                mode='lines+markers' if show_pts else 'lines',
                                name=f"{m} {t.split('(')[0]}", # Сокращаем имя в легенде
                                line=dict(width=line_w),
                                hovertemplate='<b>%{y:.2f}</b><br>%{x|%d.%m %H:%M}'
                            ))
                fig.update_layout(
                    height=chart_h, template="plotly_white",
                    legend=dict(orientation="h", y=1.02, x=0),
                    margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                    yaxis=dict(title=y_title, showgrid=True),
                    xaxis=dict(title="Время / Дата", showgrid=True)
                )
                st.plotly_chart(fig, use_container_width=True)

            # 2. СУТКИ
            with t2:
                d_g = df_v.groupby(['Date', 'Type', 'MeterID'])['Value'].sum().reset_index()
                fig_b = px.bar(d_g, x='Date', y='Value', color='Type', barmode='group', title="Суточное потребление")
                fig_b.update_layout(height=chart_h*0.8, template="plotly_white", yaxis_title="Энергия (кВт·ч / кВАр·ч)")
                st.plotly_chart(fig_b, use_container_width=True)

            # 3. КАРТА
            with t3:
                c_h1, c_h2 = st.columns(2)
                hm_m = c_h1.selectbox("Точка", meters)
                hm_t = c_h2.selectbox("Параметр", types)
                dh = df[(df['MeterID'] == hm_m) & (df['Type'] == hm_t)].copy()
                if not dh.empty:
                    dh['D'] = dh['Date'].astype(str)
                    dh['T'] = dh['Time'].astype(str)
                    fh = px.density_heatmap(dh, x='D', y='T', z='Value', nbinsy=48, color_continuous_scale='RdYlGn_r')
                    fh.update_layout(height=chart_h, yaxis=dict(autorange="reversed", title="Часы"), xaxis_title="Дата", title=f"Нагрузка: {hm_m}")
                    st.plotly_chart(fh, use_container_width=True)

            # 4. АНАЛИЗ (НОВЫЙ)
            with t4:
                st.subheader("📊 Экспертный анализ режима потребления")
                
                # Подготовка данных (только активка для начала)
                df_act = df[(df['MeterID'].isin(meters)) & (df['Suffix'] == 1)] # Актив Прием
                
                if not df_act.empty:
                    # АНАЛИЗ 1: Коэффициент заполнения (Load Factor)
                    # K = P_avg / P_max. Чем ближе к 1, тем ровнее график.
                    avg_p = df_act['Value'].mean()
                    max_p = df_act['Value'].max()
                    load_factor = avg_p / max_p if max_p > 0 else 0
                    
                    c_a1, c_a2 = st.columns(2)
                    with c_a1:
                        st.markdown(f"**Коэффициент заполнения графика ($K_{{zap}}$):** `{load_factor:.2f}`")
                        if load_factor > 0.7: st.success("✅ Отличный, ровный график нагрузки.")
                        elif load_factor > 0.4: st.info("ℹ️ Средняя неравномерность (есть пики).")
                        else: st.warning("⚠️ Очень неравномерный график! Высокие пики при малом потреблении.")
                        st.caption("Показывает эффективность использования заявленной мощности.")

                    # АНАЛИЗ 2: День / Ночь (08:00 - 20:00)
                    day_start, day_end = time(8,0), time(20,0)
                    mask_day = (df_act['Time'] >= day_start) & (df_act['Time'] < day_end)
                    day_val = df_act[mask_day]['Value'].sum()
                    night_val = df_act[~mask_day]['Value'].sum()
                    total_val = day_val + night_val
                    
                    with c_a2:
                        fig_pie = px.pie(names=['День (08-20)', 'Ночь (20-08)'], values=[day_val, night_val], 
                                         title="Распределение День/Ночь", hole=0.4)
                        fig_pie.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie, use_container_width=True)

                st.divider()
                
                # АНАЛИЗ 3: Качество (Cos Phi)
                df_calc = df[df['Suffix'].isin([1, 3])].copy()
                if not df_calc.empty:
                    piv = df_calc.pivot_table(index=['DateTime', 'MeterID'], columns='Suffix', values='Value').reset_index()
                    if 1 in piv.columns and 3 in piv.columns:
                        piv['S'] = np.sqrt(piv[1]**2 + piv[3]**2)
                        piv['CosPhi'] = np.where(piv['S'] > 0, piv[1] / piv['S'], 0)
                        
                        st.markdown("**📉 Анализ реактивной мощности (Cos φ)**")
                        
                        # График
                        fig_cos = px.line(piv, x='DateTime', y='CosPhi', color='MeterID', title="Динамика Cos φ")
                        fig_cos.add_hline(y=0.96, line_dash="dash", line_color="red", annotation_text="Норма 0.96")
                        fig_cos.update_layout(height=400, yaxis_title="Cos φ", template="plotly_white", yaxis_range=[0.5, 1.05])
                        st.plotly_chart(fig_cos, use_container_width=True)
                        
                        # Scatter Plot (Актив vs Реактив)
                        st.markdown("**Зависимость Реактива от Актива** (Позволяет выявить характер нагрузки)")
                        fig_scat = px.scatter(piv, x=1, y=3, color='MeterID', trendline="ols",
                                              labels={ "1": "Активная (кВт)", "3": "Реактивная (кВАр)" })
                        fig_scat.update_layout(height=500, template="plotly_white")
                        st.plotly_chart(fig_scat, use_container_width=True)
                        
                    else:
                        st.info("💡 Для расчета Cos φ нужны данные по Активной (код 1) и Реактивной (код 3) энергии.")
                else:
                    st.write("Загрузите файлы с данными по реактивной энергии для детального анализа качества.")

else:
    st.markdown("<h3 style='text-align: center; color: grey;'>📂 Загрузите данные для начала работы</h3>", unsafe_allow_html=True)