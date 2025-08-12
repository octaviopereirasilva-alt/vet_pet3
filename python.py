import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
import numpy as np

# Configuração da página
st.set_page_config(page_title="Painel de atendimento médico", layout="wide")

@st.cache_data
def carregar_dados():
    # Leitura com codificação correta
    df = pd.read_csv("Petiscos.csv", sep='\t', encoding='utf-8-sig')
    
    # Padronizar colunas
    df.columns = df.columns.str.strip().str.upper()
    st.write("🔍 Colunas encontradas no CSV:", df.columns.tolist())

    # Renomear colunas
    colunas_renomear = {
        "NOMEDOPET": "Pet",
        "IDADE DOPET": "Idade",
        "IDADEDOPET": "Idade",
        "GENERODOPET": "Gênero",
        "GENERODOPET": "Gênero",
        "VETERINARIO": "Medico",
        "TURNO": "Turno",
        "ATESTADO": "Atestado",
        "VETANIMAIS": "VetAnimais"
    }
    df = df.rename(columns={k: v for k, v in colunas_renomear.items() if k in df.columns})

    # Mapear valores binários com proteção
    if "Atestado" in df.columns:
        df["Atestado"] = df["Atestado"].map({"Sim": 1, "Não": 0})
    else:
        df["Atestado"] = 0

    if "VetAnimais" in df.columns:
        df["VetAnimais"] = df["VetAnimais"].map({"Sim": 1, "Não": 0})
    else:
        df["VetAnimais"] = 0

    # Simular sintomas respiratórios
    np.random.seed(42)
    df["Sindrespiratoria"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])

    return df

# Carregar os dados
df = carregar_dados()

# Título
st.title("🐾 Painel de Atendimento Médico Veterinário")

# Métricas principais
media_idade = df["Idade"].mean()
total_atestados = int(df["Atestado"].sum())
total_respiratorio = int(df["Sindrespiratoria"].sum())

st.markdown("### 📊 Resumo dos atendimentos")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Média de idade", f"{media_idade:.1f} anos")
with col2:
    st.metric("Atestados emitidos", total_atestados)
with col3:
    st.metric("Casos respiratórios", total_respiratorio)

st.divider()

# Gráficos por médico e turno
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("👨‍⚕️ Atendimentos por médico")
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x="Medico", ax=ax1, palette="coolwarm")
    ax1.set_ylabel("Quantidade")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col_b:
    st.markdown("🕒 Atendimentos por turno")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x="Turno", order=df["Turno"].value_counts().index, ax=ax2, palette="viridis")
    ax2.set_ylabel("Quantidade")
    st.pyplot(fig2)

st.divider()

# Gráficos por idade e gênero
col_c, col_d = st.columns(2)
with col_c:
    st.markdown("😷 Casos respiratórios por idade")
    resp = df[df["Sindrespiratoria"] == 1]
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    sns.histplot(resp["Idade"], bins=range(0, int(df["Idade"].max())+2), kde=True, color="purple", ax=ax3)
    ax3.set_xlabel("Idade")
    ax3.set_ylabel("Casos")
    st.pyplot(fig3)

with col_d:
    st.markdown("🐶 Distribuição por gênero")
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x="Gênero", ax=ax4, palette="pastel")
    ax4.set_ylabel("Quantidade")
    st.pyplot(fig4)

st.divider()

# Exportar CSV
st.markdown("### 📁 Exportar dados")
csv_bytes = df.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
st.download_button("Baixar CSV", data=csv_bytes, file_name='atendimentos_export.csv', mime='text/csv')

st.divider()

# Distribuição Binomial
st.markdown("### 📈 Probabilidade de atestados (Binomial)")
p_atestado = df["Atestado"].mean()
n = st.slider("Pacientes simulados (n)", 5, 50, 10)
k = st.slider("Atestados desejados (k ou mais)", 1, n, 5)

prob = 1 - binom.cdf(k - 1, n, p_atestado)
st.write(f"Taxa observada de atestados: {p_atestado:.1%}")
st.write(f"Probabilidade de ≥ {k} atestados em {n} pacientes: **{prob:.2%}**")

figb, axb = plt.subplots(figsize=(5, 3))
pmf = [binom.pmf(i, n, p_atestado) for i in range(n + 1)]
axb.bar(range(n + 1), pmf, color=["gray" if i < k else "orange" for i in range(n + 1)])
axb.set_xlabel("Número de atestados")
axb.set_ylabel("Probabilidade")
st.pyplot(figb)

st.divider()

# Distribuição de Poisson
st.markdown("### 📈 Casos respiratórios por turno (Poisson)")
media_turno = df.groupby("Turno")["Sindrespiratoria"].sum().mean()
k_p = st.slider("Casos respiratórios desejados (k ou mais)", 0, 10, 3)

prob_p = 1 - poisson.cdf(k_p - 1, media_turno)
st.write(f"Média de casos respiratórios por turno: **{media_turno:.2f}**")
st.write(f"Probabilidade de ≥ {k_p} casos: **{prob_p:.2%}**")

figp, axp = plt.subplots(figsize=(5, 3))
poisson_pmf = [poisson.pmf(i, media_turno) for i in range(0, 11)]
axp.bar(range(11), poisson_pmf, color=["gray" if i < k_p else "orange" for i in range(11)])
axp.set_xlabel("Número de casos")
axp.set_ylabel("Probabilidade")
st.pyplot(figp)
