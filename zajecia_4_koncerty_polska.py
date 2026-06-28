#!/usr/bin/env python
# coding: utf-8

# # Wizualizacja rynku koncertów muzycznych w Polsce
# 
# Notebook do zadania z zajęć 4: wczytanie danych, eksploracja oraz zestaw interaktywnych wykresów w Plotly.

# ## Import bibliotek

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "plotly_white"


# ## Generowanie danych
# 
# Poniższy kod generuje plik `koncerty_polska.csv` zgodnie z treścią zadania. Uruchamiamy go raz, a potem dalej pracujemy już na zapisanym pliku CSV.

# In[2]:


np.random.seed(42)

n = 1200

miasta = {
    "Warszawa":   (52.2297, 21.0122, 1.00),
    "Kraków":     (50.0647, 19.9450, 0.75),
    "Wrocław":    (51.1079, 17.0385, 0.65),
    "Poznań":     (52.4064, 16.9252, 0.55),
    "Gdańsk":     (54.3520, 18.6466, 0.55),
    "Łódź":       (51.7592, 19.4560, 0.50),
    "Katowice":   (50.2649, 19.0238, 0.45),
    "Lublin":     (51.2465, 22.5684, 0.30),
    "Białystok":  (53.1325, 23.1688, 0.25),
    "Szczecin":   (53.4285, 14.5528, 0.35),
}

gatunki = ["rock", "pop", "hip-hop", "electronic", "jazz",
           "classical", "folk", "metal", "indie", "reggae"]

typy_obiektow = ["klub", "arena", "stadion", "festiwal", "teatr", "amfiteatr"]
kapacjety = {
    "klub": (200, 1500), "arena": (3000, 15000), "stadion": (20000, 70000),
    "festiwal": (10000, 80000), "teatr": (400, 2000), "amfiteatr": (1500, 8000),
}
cena_bazowa = {
    "rock": 150, "pop": 200, "hip-hop": 180, "electronic": 160, "jazz": 130,
    "classical": 110, "folk": 90, "metal": 140, "indie": 100, "reggae": 110,
}
cena_mnoznik = {
    "klub": 0.7, "arena": 1.3, "stadion": 1.8,
    "festiwal": 1.5, "teatr": 1.2, "amfiteatr": 1.0,
}

start_date = datetime(2024, 1, 1)
daty = [start_date + timedelta(days=int(d)) for d in np.random.randint(0, 730, n)]

wagi = np.array([miasta[m][2] for m in miasta])
miasto = np.random.choice(list(miasta.keys()), n, p=wagi / wagi.sum())
gatunek = np.random.choice(gatunki, n)
typ_obiektu = np.random.choice(
    typy_obiektow,
    n,
    p=[0.40, 0.15, 0.05, 0.10, 0.15, 0.15]
)

pojemnosc = np.array([np.random.randint(*kapacjety[t]) for t in typ_obiektu])
wypelnienie_losowe = np.clip(np.random.beta(5, 2, n), 0.15, 1.0)
sprzedane = (pojemnosc * wypelnienie_losowe).astype(int)

cena = np.array([cena_bazowa[g] * cena_mnoznik[t] for g, t in zip(gatunek, typ_obiektu)])
cena = np.round(cena * np.random.uniform(0.7, 1.4, n), -1)
przychod = (cena * sprzedane).astype(int)

df = pd.DataFrame({
    "event_id": range(50001, 50001 + n),
    "data": daty,
    "miasto": miasto,
    "latitude": [miasta[m][0] for m in miasto],
    "longitude": [miasta[m][1] for m in miasto],
    "gatunek": gatunek,
    "typ_obiektu": typ_obiektu,
    "pojemnosc": pojemnosc,
    "bilety_sprzedane": sprzedane,
    "cena_biletu_pln": cena,
    "przychod_pln": przychod,
})

df.to_csv("koncerty_polska.csv", index=False)
print(f"Wygenerowano plik 'koncerty_polska.csv' — {len(df)} koncertów")


# ## Część 1 — Wczytanie i wstępna eksploracja

# In[3]:


df = pd.read_csv("koncerty_polska.csv", parse_dates=["data"])

print("Shape:", df.shape)
print("\nUnikalne miasta:", df["miasto"].nunique())
print("Unikalne gatunki:", df["gatunek"].nunique())


# In[4]:


df.head()


# In[5]:


df.dtypes


# ## Część 2 — Wykres słupkowy
# 
# Łączny przychód w każdym mieście, posortowany malejąco.

# In[6]:


przychod_miasto = (
    df.groupby("miasto", as_index=False)["przychod_pln"]
      .sum()
      .sort_values("przychod_pln", ascending=False)
)

fig = px.bar(
    przychod_miasto,
    x="miasto",
    y="przychod_pln",
    title="Łączny przychód z koncertów według miasta",
    labels={
        "miasto": "Miasto",
        "przychod_pln": "Łączny przychód [PLN]"
    },
    text_auto=",.0f"
)
fig.update_layout(
    xaxis_title="Miasto",
    yaxis_title="Łączny przychód [PLN]",
    showlegend=False,
    height=500
)
fig.update_traces(
    hovertemplate="<b>%{x}</b><br>Przychód: %{y:,.0f} PLN<extra></extra>"
)
fig.show()


# ## Część 3 — Wykres liniowy / szereg czasowy
# 
# Najpierw agregacja do poziomu miesiąca, a potem dwa wykresy liniowe: ogólna liczba koncertów oraz liczba koncertów z podziałem na typ obiektu.

# In[7]:


df["miesiac"] = df["data"].dt.to_period("M").astype(str)

koncerty_miesiac = (
    df.groupby("miesiac", as_index=False)
      .size()
      .rename(columns={"size": "liczba_koncertow"})
)

fig = px.line(
    koncerty_miesiac,
    x="miesiac",
    y="liczba_koncertow",
    markers=True,
    title="Liczba koncertów w kolejnych miesiącach",
    labels={
        "miesiac": "Miesiąc",
        "liczba_koncertow": "Liczba koncertów"
    }
)
fig.update_layout(height=500)
fig.update_traces(
    hovertemplate="Miesiąc: %{x}<br>Liczba koncertów: %{y}<extra></extra>"
)
fig.show()


# In[8]:


koncerty_miesiac_typ = (
    df.groupby(["miesiac", "typ_obiektu"], as_index=False)
      .size()
      .rename(columns={"size": "liczba_koncertow"})
)

fig = px.line(
    koncerty_miesiac_typ,
    x="miesiac",
    y="liczba_koncertow",
    color="typ_obiektu",
    markers=True,
    title="Miesięczna liczba koncertów według typu obiektu",
    labels={
        "miesiac": "Miesiąc",
        "liczba_koncertow": "Liczba koncertów",
        "typ_obiektu": "Typ obiektu"
    }
)
fig.update_layout(height=550, legend_title_text="Typ obiektu")
fig.show()


# ## Część 4 — Histogram i boxplot
# 
# Dla histogramu wybrałem `nbins=50`, bo przy 20 binach rozkład jest zbyt ogólny, a przy 100 robi się zbyt poszarpany.

# In[9]:


fig = px.histogram(
    df,
    x="cena_biletu_pln",
    nbins=50,
    title="Rozkład cen biletów na koncerty",
    labels={"cena_biletu_pln": "Cena biletu [PLN]"}
)
fig.update_layout(
    xaxis_title="Cena biletu [PLN]",
    yaxis_title="Liczba koncertów",
    height=500
)
fig.update_traces(
    hovertemplate="Cena: %{x} PLN<br>Liczba koncertów: %{y}<extra></extra>"
)
fig.show()


# In[10]:


fig = px.box(
    df,
    x="typ_obiektu",
    y="przychod_pln",
    title="Rozkład przychodu według typu obiektu",
    labels={
        "typ_obiektu": "Typ obiektu",
        "przychod_pln": "Przychód [PLN]"
    }
)
fig.update_layout(height=550)
fig.show()


# **Komentarz:** Najwyższe przychody generują przede wszystkim stadiony i festiwale. Wynika to z ich bardzo dużej pojemności — nawet przy podobnym poziomie wypełnienia liczba sprzedanych biletów jest dużo większa niż w klubach, teatrach czy amfiteatrach.

# ## Część 5 — Scatter plot z kodowaniem koloru
# 
# Dodajemy kolumnę `wypelnienie`, czyli udział sprzedanych biletów w pojemności obiektu.

# In[11]:


df["wypelnienie"] = df["bilety_sprzedane"] / df["pojemnosc"]

df[["bilety_sprzedane", "pojemnosc", "wypelnienie"]].head()


# In[12]:


fig = px.scatter(
    df,
    x="cena_biletu_pln",
    y="wypelnienie",
    color="gatunek",
    size="pojemnosc",
    hover_data=["miasto", "typ_obiektu"],
    title="Cena biletu a wypełnienie obiektu",
    labels={
        "cena_biletu_pln": "Cena biletu [PLN]",
        "wypelnienie": "Wypełnienie obiektu",
        "gatunek": "Gatunek",
        "pojemnosc": "Pojemność"
    },
    size_max=45
)
fig.update_layout(height=600)
fig.update_yaxes(tickformat=".0%")
fig.show()


# **Komentarz:** Nie widać mocnej zależności między ceną biletu a wypełnieniem obiektu. Punkty są dość rozproszone, a współczynnik korelacji jest bliski zeru, więc w tych danych wyższa cena biletu nie oznacza automatycznie niższego ani wyższego wypełnienia sali.

# In[13]:


korelacja = df["cena_biletu_pln"].corr(df["wypelnienie"])
print(f"Korelacja ceny biletu i wypełnienia: {korelacja:.3f}")


# ## Część 6 — Mapa
# 
# Agregacja do poziomu miasta: średnia cena biletu, liczba koncertów, łączny przychód oraz współrzędne geograficzne.

# In[14]:


miasta_aggr = (
    df.groupby("miasto", as_index=False)
      .agg(
          latitude=("latitude", "first"),
          longitude=("longitude", "first"),
          srednia_cena_biletu=("cena_biletu_pln", "mean"),
          liczba_koncertow=("event_id", "count"),
          laczny_przychod=("przychod_pln", "sum")
      )
)

miasta_aggr


# In[15]:


fig = px.scatter_mapbox(
    miasta_aggr,
    lat="latitude",
    lon="longitude",
    size="liczba_koncertow",
    color="srednia_cena_biletu",
    hover_name="miasto",
    hover_data={
        "liczba_koncertow": True,
        "srednia_cena_biletu": ":.2f",
        "laczny_przychod": ":,.0f",
        "latitude": False,
        "longitude": False,
    },
    color_continuous_scale="RdBu_r",
    zoom=5,
    center={"lat": 52, "lon": 19},
    mapbox_style="open-street-map",
    height=600,
    title="Rynek koncertów w polskich miastach"
)
fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
fig.show()


# ## Część 7 — Kompozycja: subploty 2×2
# 
# Cztery różne wykresy podsumowujące zbiór w jednej figurze.

# In[16]:


# Dane pomocnicze do subplotów
przychod_miasto = (
    df.groupby("miasto", as_index=False)["przychod_pln"]
      .sum()
      .sort_values("przychod_pln", ascending=False)
)

koncerty_gatunek = (
    df.groupby("gatunek", as_index=False)
      .size()
      .rename(columns={"size": "liczba_koncertow"})
      .sort_values("liczba_koncertow", ascending=False)
)

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Przychód wg miasta",
        "Liczba koncertów wg gatunku",
        "Histogram cen biletów",
        "Wypełnienie wg typu obiektu"
    )
)

# 1. Słupki przychodu wg miasta
fig.add_trace(
    go.Bar(
        x=przychod_miasto["miasto"],
        y=przychod_miasto["przychod_pln"],
        name="Przychód wg miasta",
        hovertemplate="%{x}<br>Przychód: %{y:,.0f} PLN<extra></extra>"
    ),
    row=1,
    col=1
)

# 2. Słupki liczby koncertów wg gatunku
fig.add_trace(
    go.Bar(
        x=koncerty_gatunek["gatunek"],
        y=koncerty_gatunek["liczba_koncertow"],
        name="Liczba koncertów wg gatunku",
        hovertemplate="%{x}<br>Liczba koncertów: %{y}<extra></extra>"
    ),
    row=1,
    col=2
)

# 3. Histogram cen
fig.add_trace(
    go.Histogram(
        x=df["cena_biletu_pln"],
        nbinsx=50,
        name="Ceny biletów",
        hovertemplate="Cena: %{x} PLN<br>Liczba: %{y}<extra></extra>"
    ),
    row=2,
    col=1
)

# 4. Boxplot wypełnienia wg typu obiektu
for typ in sorted(df["typ_obiektu"].unique()):
    fig.add_trace(
        go.Box(
            y=df.loc[df["typ_obiektu"] == typ, "wypelnienie"],
            name=typ,
            boxmean=True,
            hovertemplate=f"Typ obiektu: {typ}<br>Wypełnienie: %{{y:.2%}}<extra></extra>"
        ),
        row=2,
        col=2
    )

fig.update_layout(
    title_text="Mini-dashboard rynku koncertów w Polsce",
    height=850,
    showlegend=False,
    bargap=0.2
)

fig.update_xaxes(title_text="Miasto", row=1, col=1)
fig.update_yaxes(title_text="Przychód [PLN]", row=1, col=1)
fig.update_xaxes(title_text="Gatunek", row=1, col=2)
fig.update_yaxes(title_text="Liczba koncertów", row=1, col=2)
fig.update_xaxes(title_text="Cena biletu [PLN]", row=2, col=1)
fig.update_yaxes(title_text="Liczba koncertów", row=2, col=1)
fig.update_xaxes(title_text="Typ obiektu", row=2, col=2)
fig.update_yaxes(title_text="Wypełnienie", tickformat=".0%", row=2, col=2)

fig.show()


# ## Część 8 — Wnioski
# 
# 1. Największym rynkiem koncertowym w danych jest **Warszawa** — ma najwięcej koncertów oraz najwyższy łączny przychód.
# 2. Wysokie przychody generują głównie **stadiony** i **festiwale**, bo mają największą pojemność i sprzedają najwięcej biletów.
# 3. Najwyższe średnie ceny biletów występują w obiektach typu **stadion**, a następnie **festiwal** i **arena**.
# 4. Nie widać wyraźnej zależności między ceną biletu a wypełnieniem obiektu — korelacja jest bardzo niska, więc cena sama w sobie nie tłumaczy frekwencji.
# 5. W szeregu miesięcznym widać zmienność liczby koncertów, ale sezonowość nie jest bardzo jednoznaczna. Najwięcej koncertów wypada w **październiku 2025**, a najmniej w **sierpniu 2025**.
