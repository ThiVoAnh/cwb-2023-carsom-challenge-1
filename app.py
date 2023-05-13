#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import streamlit as st
import joblib


st.set_page_config(page_title = 'Car some price prediction page', page_icon=None, layout='centered', initial_sidebar_state='auto')

@st.cache(allow_output_mutation=True)
def load(isolationforest_path, label_encoder_path, dec_model_path):
    isolationforest = joblib.load(isolationforest_path)
    label_encoder = joblib.load(label_encoder_path)
    dec_model = joblib.load(dec_model_path)
    return isolationforest , label_encoder, dec_model

def inference(row, cols, isolationforest, label_encoder, dec_model):
    df = pd.DataFrame([row], columns = cols)

    # Drop the columns that will not be utilized
    # drop_cols = ['lead_id', 'marketplace_id', 'marketplace_car_id', 'used_dealer_company_id', 'dealer_id']
    # df = df.drop(drop_cols, axis = 1)

    # Drop the Null values
    # df = df.dropna()

    # Remove outlier
    # isolationforest.fit(df[['reserveprice']].values)
    # df['score'] = isolationforest.decision_function(df['reserveprice'].values)
    # df['anomaly']=isolationforest.predict(df[['reserveprice']].values)

    # car_df_outlier_rm = df.loc[df['anomaly'] == 1]
    # car_df_outlier_rm = car_df_outlier_rm.drop(['score', 'anomaly'], axis = 1)

    # Label encoder
    obj_feats = (df.dtypes == 'object')
    obj_cols = list(obj_feats[obj_feats].index)
    df_label = df.copy()
    for col in obj_cols:
        df_label[col] = label_encoder.fit_transform(df_label[col])

    

    # Predict price
    price = dec_model.predict(df_label)[0]

    return price

st.title('Car some price prediction page')
st.write('Prediction Price of a Car')
st.write('Please fill in the details')

car_brand = st.sidebar.selectbox("Brand", ('honda', 'suzuki', 'daihatsu', 'ford', 'isuzu', 'toyota',
       'mitsubishi', 'chevrolet', 'mazda', 'nissan', 'wuling', 'kia',
       'volkswagen', 'bmw', 'land rover', 'hyundai', 'peugeot',
       'mercedes-benz', 'mini', 'subaru', 'datsun', 'audi', 'ota',
       'renault', 'tata', 'fordta'))
car_model = st.sidebar.selectbox("Model", ('freed', 'city', 'hr-v', 'ertiga', 'sigra', 'fiesta', 'brio',
       'mu-x', 'cr-v', 'br-v', 'calya', 'swift', 'xenia', 'xpander',
       'outlander', 'alphard', 'outlander sport', 'civic', 'avanza',
       'trax', 'delica', '8', 'mobilio', 'sienta', 'cx-30', 'jazz',
       'livina', 'grand livina', 'vellfire', 'fortuner', 'camry', 'juke',
       'gran max', 'terios', 'march', 'pajero sport', 'confero',
       'kijang innova', 'yaris', 'elysion', 'biante', 'sedona', 'polo',
       'cx-5', '3', 'trailblazer', 'rush', 'vios', 'panther', 'harrier',
       'triton', 'innova', 'pajero', 'almaz', 'range rover evoque',
       'tiguan', 'cx-9', 'x-over', 'voxy', 'grand avega', '207', 'serena',
       'x3', 'c-hr', 'mu-7', 'apv', '2', 'karimun', '6', 'terra',
       'kijang', 'x1', 'x-trail', 'cortez', 'limo', 'h-1', 'b', 'captiva',
       'rio', 'sx4', 'everest', 'ayla', 'kijang super', 'c', 'gle', 'e',
       'hilux', 'agya', 'carry', 'cx-3', 'cooper countryman', 'nav1',
       'ml', 'corolla altis', 'tucson', 'grand vitara', 'accord',
       'corolla', 'cla', 'dyna', 'eclipse cross', 'ranger', 'teana',
       'land cruiser', 'xv', 'colt l300', '5', 'i20', 'lancer', 'slk',
       'ignis', 'cr-z', 'raize', 'luxio', 'go panca', 'elgrand', 'sirion',
       'seltos', 'xl7', 'evalia', 'splash', 'grand escudo', 'ecosport',
       'corolla cross', 'go+ panca', 'etios', 'cx-7', 'gl',
       'cooper paceman', 'picanto', 'elf', 'cooper', 'grandis', 'colt fe',
       'sorento', 'etios valco', 'a6', 'altis', 'aveo', 'katana', 'x5',
       'colt t120ss', 'h-4', 'range roverer', 'spin', 'baleno', 'terrano',
       'c-trail', 'triber', 'stream', 'mirage', 'wish', 'hiace', 'a4',
       'traga', 'santa fe', 'mu-x6', 'a7', 'aerio', 'odyssey', 'wber',
       'sportage', 'gla', 'ga', 'colt l200', 'colt l60', 'trooper', 'k',
       'cross', 'kwid', 'x4', 'grand i10', 'cx-50', 'colt l309', 'taruna',
       'glo', '4', 'sotana', 't livina', 'traxy', 'h-7', 'xv sport',
       '200e', '230e', '1', 'so', 'hr', '--v'))
car_variant = st.sidebar.selectbox("Variant", ('s', 'vtec', 'gl', 'gx', 'm', 'rs', 'r2', '2wd', 'e', 'g', 'x',
       'ultimate', 'gls', 'px', 'fd1', 'r std', 'turbo ltz', 'royal',
       'no variant', 'satya e', 'gt', 'el', 'sv', 'z', 'g trd', 'v', 'rx',
       'd', 'tx', 'xs', 'dakar', 'i-dsi', 's l', 's ltd', 'i-vtec',
       'skyactiv', 'veloz', 'gt tsi', 'xv', 'touring', '20i (ckd)', 'ltz',
       'prestige', 's act lux', 'exceed', 'grand touring', '240 g',
       'dc gls 4x4', 'vrz', 'sport', 'v diesel', 'd exceed',
       'mugen prestige', 'vrz 4x2', 'lt lux + sc cvt', 'g-n140', 'tsi',
       'v lux', 'e plus', 'highway star', 'e prestige', 'dreza', '18i',
       'hybrid', 'reborn g', 'xdrive20i (ckd)', 'j', 'sport gt',
       'single tone', 'tfr85hg', 'luxury', '20i', 'dx', 's trd sportivo',
       'box', 'vl', 'r', 'lgx', 'sdrive18i', 'ls', 'st autech', 'c t lux',
       'tsi allspace', 'crdi', '200 urban', 'reborn v', 'se', 'turbo',
       'tc e', 's-cross', 'kf 42 short', '200 cgi (ckd)',
       '400 4matic (ckd)', 'l', '300 (ckd)', 'double cabin g 4x4',
       'fd pick up', '280', 'dakar ultimate 4x2', 'sc', 'vrz trd',
       'dakar 4x2', 't lux', 'trd sportivo', '330 d', 'x-gear',
       'sgx luxury', 'xdrive20i xline', '20i e90 (ckd)', 'r sporty',
       'vrz 4x4', '300', 'q-n140', 's trd heykers', 'jlx', 'vti-l',
       'g lux', '240 elegance', 'turbo prestige', '200', '110 st',
       '200 (ckd)', 'i', 'lt', 'gl arena', 'base', 'a', 'pick up',
       'tx adventure', 's (ckd)', '250 xv', 'turbo premier', 'g-h30',
       '180', 'hv dual tone', 'ts extra', 'rx red edition', 'prado (4wd)',
       'reborn venturer diesel', 'panoramic', 'high', 'i (awd)',
       '23i (ckd)', 'gx arena', 'xls', 'x elegant', '200 cgi', 'glxi',
       'prestige special edition', '250 amg', 'vti', 'l lux', 'li', 'xi',
       'gl sporty', 'single cabin', 's trd', 'x dlx', 'srz', 'zf1', 'rc1',
       'gr sport', 'v-limited', 't', 'm804rs', 'v-extra', 'ex+', 'q',
       'st', 'ls turbo', 'beta', 'limited edition', 'r dlx', 'dc hdx 4x4',
       '3-d', 'm602rs', '23i', 'v6', 'dakar 4x4', 'cross premium', 'xl-7',
       '2wd ckd', 'lm', 'x std', 'g trd-luxury', 'gt3', 'titanium',
       'q venturer', 'g diesel', 'prado', 'alpha', 'zdi hybrid', 'xg',
       'xt', 'low', '200 k', 'd exceed 4x2', 'gs', 'zg', '320',
       '500 (ckd)', 'comfort touring', 'autech', 'd exceed 4x4', 'sg ne',
       'li deluxe', '250', '71 diesel (2wd)', 'gt skyactiv', 'turbo hb e',
       'st 4x2', 'xi family', 'tfsi (ckd)', '250 d', 'hpe 4x4',
       'gx short', 'r skyactiv', 'kf 40 short', 'xi sporty', '25i',
       's luxury', '20i e30', 'pick up fd ps', 'comfort touring b e',
       'estilo', 'cygnus', 'tfsi', 'cross', '250 cgi',
       'dakar 4x2 limited edition', 'e lux', 'r family', 'gt2',
       'd minibusryms', 'hatchback', 'sgx', 'gx ags', '100 (ckd)',
       'dc hdx 4x2', 'gl mt', 'xlt', 's trd sportivo ultimo',
       'vrz gr sport', 'gl elegant', 'blind van', '270 short', 'm dlx',
       'd std', 'trend', 'd fms', 'rs hatchback', 'rxz',
       'dakar ultimate 4x4', 'glx', '170', 'sgx arena', 'tc (ckd)', 'ts',
       '200 amg', 'd elegant', 'c', '1840', 'v prestige', 'extra x',
       'commuter', '250 g', 'reborn venturer gasoline', 'diesel 304',
       'l cbu', 'lx', 'd fmc', 'g lux-luxury', 'r attivo', 've', '10i',
       'v-gear', 'kf 49 short', 'r diesel crdi', 'cross over',
       'd fmc prestige', 'gc415v dlx', 's c', 't st', 'sport a 4x2',
       'pick up box', 'm704rs', '4wd', 'touring extra', 'v2', 'krista',
       '230 classi', 'r250', 'c c t lux', 'satya s', 'rx ckd', 'e-extra',
       '3wd', 'highway star autech', 'highway star arena', 's sporty',
       'sdrive18i (ckd)', 'tx dlx', 'gs deluxe', '330', 'turbo-trail',
       'cross skyactiv', 'r ltd', 'd-gear', 'icken', 'd minibus',
       'd fmc gls', 'climber', 'sc-h30', 'tsi autech', 'e altpace', 'sx',
       'royal adventure', '245 elegance', 'rs-cross', '2wd ckd deluxe',
       'g-e140', '2927640 cgi', 'royal extra', 'e diesel', 'tone',
       '28i (ckd)', 'd sporty', 'vrz lux', 'grand touring prestige',
       'zf7', 'cx deluxe', 'ch prestige', 's skyactiv', '330 classic',
       't std', 'dakar 4x3', 'sport gt ags', 'e trd', '100', 'gl ags',
       'gs ags', 't prestige', 'double cabin g 4x2',
       'trd sportivo ultimo', 'gl 29017', 'comfort', 'g arena',
       'royal hatchback', 's cross', 'rx-cross', 'gt8', '250 d short',
       'm702rs', 'd extra', '1840 amg', 'tsi trd', 'ssx', '20i e10 (ckd)',
       '20i e', 'sport prestige', 'gx deluxe', '2 st', '25i e90 (ckd)',
       '280 7g tronic', '320 (ckd)', 's short', 'c t lux + sc cvt', 'zga',
       '400 (ckd)', 'x-extra', '20i e60 (ckd)', 's e', '230 (ckd)',
       'xi deluxe', 'vrz 4x2 limited edition', 'm804rs sportivo', 'ge',
       '2', 'gl autech', '240', 'xi dlx', 'sport gt touring',
       'd exceed 4x6', '4wd ckd', '5wd'))
car_engine = st.slider("Engine", min_value = 1.0, max_value = 3.8, step = 0.1)
car_year = st.slider("Year",min_value = 1990, max_value = 2023, step = 1)
car_transmission = st.sidebar.selectbox("Transmission", ('Auto', 'Manual'))
row = [car_brand, car_model, car_variant, car_engine, car_year, car_transmission]
cols = ['car_brand', 'car_model', 'car_variant', 'car_engine', 'car_year', 'car_transmission']

if (st.button('Estimate Car Price')):
    isolationforest, label_encoder, dec_model = load('isolationforest.pkl', 'label_encoder.pkl', 'dec_model.pkl')
    result = inference(row, cols, isolationforest, label_encoder, dec_model)
    st.write(result)