import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

# Constants
MIN_VAL = 0.0
MAX_VAL = 100000000.0
HEATMAP_SIZE = 10

# Define some reused functions
def getNormalCDF(x):
    return norm.cdf(x, loc=0, scale=1)

def getNormalPDF(x):
    return norm.pdf(x, loc=0, scale=1)

def getDPlus(volatility, strikePrice, spotPrice, timeToMaturity, interestRate):
    return (1 / (volatility * math.sqrt(timeToMaturity))) * (np.log(spotPrice / strikePrice) + timeToMaturity * (interestRate + volatility ** 2 / 2))

def getDMinus(volatility, strikePrice, spotPrice, timeToMaturity, interestRate):
    return getDPlus(volatility, strikePrice, spotPrice, timeToMaturity, interestRate) - volatility * math.sqrt(timeToMaturity)

def getCallPrice(volatility, strikePrice, spotPrice, timeToMaturity, interestRate):
    dPlus = getDPlus(volatility, strikePrice, spotPrice, timeToMaturity, interestRate)
    dMinus = getDMinus(volatility, strikePrice, spotPrice, timeToMaturity, interestRate)
    return getNormalCDF(dPlus) * spotPrice - getNormalCDF(dMinus) * strikePrice * np.exp(-interestRate * timeToMaturity)

def getPutPrice(volatility, strikePrice, spotPrice, timeToMaturity, interestRate):
    dPlus = getDPlus(volatility, strikePrice, spotPrice, timeToMaturity, interestRate)
    dMinus = getDMinus(volatility, strikePrice, spotPrice, timeToMaturity, interestRate)
    return getNormalCDF(-dMinus) * strikePrice * np.exp(-interestRate * timeToMaturity) - getNormalCDF(-dPlus) * spotPrice


# Page Setup
st.title("Options Pricing - Interactive Heatmap")

st.write("---")

st.sidebar.header("Black-Scholes Option Pricing Model", divider="red")

st.sidebar.subheader("Input Configuration")

# Get user input
S = st.sidebar.number_input("Underlying Asset Price", MIN_VAL, MAX_VAL, value=79.93, step=0.01)
K = st.sidebar.number_input("Strike Price", MIN_VAL, MAX_VAL, value=80.0, step=0.01)
T = st.sidebar.number_input("Time to Maturity in Months", MIN_VAL, MAX_VAL, value=1.10, step=0.01)
volatility = st.sidebar.number_input("Volatility", MIN_VAL, MAX_VAL, value=0.25, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate", MIN_VAL, MAX_VAL, value=0.10, step=0.01)

st.sidebar.write("---")

st.sidebar.subheader("Heatmap Configuration")

minSpot = st.sidebar.number_input("Min Spot Price", MIN_VAL, MAX_VAL, value=49.94, step=0.01)
maxSpot = st.sidebar.number_input("Max Spot Price", MIN_VAL, MAX_VAL, value=109.92, step=0.01)

minVol = st.sidebar.number_input("Min Volatility", MIN_VAL, MAX_VAL, value=0.10, step=0.01)
maxVol = st.sidebar.number_input("Max Volatility", MIN_VAL, MAX_VAL, value=0.40, step=0.01)

col1, col2 = st.columns(2)

# Heatmap Data Calculations
spotStep = (maxSpot - minSpot) / (HEATMAP_SIZE-1)
baseSpot = np.arange(minSpot, maxSpot + spotStep, spotStep)
spotGrid = np.tile(baseSpot, (HEATMAP_SIZE, 1))

volStep = (maxVol - minVol) / (HEATMAP_SIZE-1)
baseVol = np.arange(maxVol, minVol - volStep, -volStep)
volGrid = np.tile(baseVol.reshape(-1, 1), (1, HEATMAP_SIZE))

callPrices = getCallPrice(volatility=volGrid, strikePrice=K, spotPrice=spotGrid, timeToMaturity=T, interestRate=r)
putPrices = getPutPrice(volatility=volGrid, strikePrice=K, spotPrice=spotGrid, timeToMaturity=T, interestRate=r)

# CALL COLUMN
callPrice = getCallPrice(volatility=volatility, strikePrice=K, spotPrice=S, timeToMaturity=T, interestRate=r)
col1.write(f"Call Price: {callPrice:.2f}")

col1.text("Call Price Heatmap")

call_fig, call_ax = plt.subplots()
call_im = call_ax.imshow(callPrices)
call_ax.set_xticks(np.arange(len(baseSpot)))
call_ax.set_xticklabels(np.round(baseSpot, 2))
call_ax.set_yticks(np.arange(len(baseVol)))
call_ax.set_yticklabels(np.round(baseVol, 2))

call_cbar = call_fig.colorbar(call_im, ax=call_ax)
call_cbar.ax.set_ylabel("Call Option Price", rotation=-90, va="bottom")

for i in range(HEATMAP_SIZE):
    for j in range(HEATMAP_SIZE):
        text = call_ax.text(j, i, f"{callPrices[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

col1.pyplot(call_fig)

# PUT COLUMN
putPrice = getPutPrice(volatility=volatility, strikePrice=K, spotPrice=S, timeToMaturity=T, interestRate=r)
col2.write(f"Put Price: {putPrice:.2f}")

col2.text("Put Price Heatmap")

put_fig, put_ax = plt.subplots()
put_im = put_ax.imshow(putPrices)
put_ax.set_xticks(np.arange(len(baseSpot)))
put_ax.set_xticklabels(np.round(baseSpot, 2))
put_ax.set_yticks(np.arange(len(baseVol)))
put_ax.set_yticklabels(np.round(baseVol, 2))

put_cbar = put_fig.colorbar(put_im, ax=put_ax)
put_cbar.ax.set_ylabel("Put Option Price", rotation=-90, va="bottom")

for i in range(HEATMAP_SIZE):
    for j in range(HEATMAP_SIZE):
        text = put_ax.text(j, i, f"{putPrices[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

col2.pyplot(put_fig)


# Greeks Section
st.header("Greeks")

# Greeks Calculations

dPlus = getDPlus(volatility=volatility, strikePrice=K, spotPrice=S, timeToMaturity=T, interestRate=r)
dMinus = getDMinus(volatility=volatility, strikePrice=K, spotPrice=S, timeToMaturity=T, interestRate=r)

callDelta = getNormalCDF(dPlus)
putDelta = callDelta - 1
callGamma = getNormalPDF(dPlus) / (S * volatility * math.sqrt(T))
putGamma = callGamma
callVega = S * math.sqrt(T) * getNormalPDF(dPlus)
putVega = callVega
callTheta = -(S * getNormalPDF(dPlus) * volatility) / (2 * math.sqrt(T)) - r * K * np.exp(-r * T) * getNormalCDF(dMinus)
putTheta = -(S * getNormalPDF(dPlus) * volatility) / (2 * math.sqrt(T)) + r * K * np.exp(-r * T) * getNormalCDF(-dMinus)
callRho = K * T * np.exp(-r * T) * getNormalCDF(dMinus)
putRho = -K * T * np.exp(-r * T) * getNormalCDF(-dMinus)

greeks = pd.DataFrame({
    "": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    "Call": [callDelta, callGamma, callVega, callTheta, callRho],
    "Put": [putDelta, putGamma, putVega, putTheta, putRho]
})

greeks.set_index("", inplace=True)

st.table(greeks)