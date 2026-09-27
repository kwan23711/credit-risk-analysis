
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Analysis Prototype",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("final_model.joblib")


model = load_model()


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

numerical_cols = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]


# ============================================================
# PROJECT RESULTS
# ============================================================

test_metrics = {
    "Accuracy": 0.8957,
    "Precision": 0.7584,
    "Recall": 0.7680,
    "F1-Score": 0.7631,
    "Balanced Accuracy": 0.8497
}


cv_test_df = pd.DataFrame({
    "Metric": [
        "Recall",
        "Precision",
        "F1",
        "Balanced Accuracy",
        "Accuracy"
    ],
    "Cross Validation": [
        0.7646,
        0.7468,
        0.7555,
        0.8460,
        0.8918
    ],
    "Test Set": [
        0.7680,
        0.7584,
        0.7631,
        0.8497,
        0.8957
    ]
})

cv_test_df["Difference"] = (
    cv_test_df["Test Set"]
    - cv_test_df["Cross Validation"]
)


model_results = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Decision Tree",
        "Logistic Regression",
        "Logistic Regression",
        "Random Forest",
        "Random Forest"
    ],

    "SMOTENC": [
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "No"
    ],

    "Recall": [
        0.7716,
        0.7646,
        0.7570,
        0.5553,
        0.7470,
        0.7196
    ],

    "Precision": [
        0.6802,
        0.7468,
        0.5615,
        0.7652,
        0.7901,
        0.9678
    ],

    "F1": [
        0.7230,
        0.7555,
        0.6447,
        0.6434,
        0.7679,
        0.8254
    ],

    "Balanced Accuracy": [
        0.8350,
        0.8460,
        0.7958,
        0.7538,
        0.8457,
        0.8565
    ],

    "Accuracy": [
        0.8707,
        0.8918,
        0.8176,
        0.8655,
        0.9012,
        0.9334
    ]
})


# ============================================================
# GLOBAL SHAP RESULTS
# ============================================================

global_shap = pd.DataFrame({
    "Feature": [
        "loan_percent_income",
        "loan_grade",
        "person_home_ownership",
        "loan_intent",
        "loan_int_rate",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "person_age",
        "cb_person_cred_hist_length",
        "cb_person_default_on_file"
    ],

    "Mean |SHAP|": [
        0.1036282779910512,
        0.08406214897268738,
        0.08319477231582673,
        0.08096175371990708,
        0.07895536090794897,
        0.07101999703060105,
        0.02748041405520047,
        0.020671480424605187,
        0.015563776256840661,
        0.01006998930153001,
        0.004226689508739867
    ]
})


# ============================================================
# FEATURE RANK COMPARISON
# ============================================================

rank_df = pd.DataFrame({
    "Feature": [
        "loan_percent_income",
        "loan_grade",
        "person_home_ownership",
        "loan_int_rate",
        "person_income",
        "loan_intent",
        "cb_person_default_on_file",
        "person_emp_length",
        "loan_amnt",
        "person_age",
        "cb_person_cred_hist_length"
    ],

    "Stat Rank": [
        2, 3, 5, 1, 4, 8, 6, 7, 9, 10, 11
    ],

    "Built-in Rank": [
        1, 6, 4, 3, 2, 5, 11, 7, 8, 9, 10
    ],

    "Permutation Rank": [
        1, 4, 2, 6, 3, 5, 11, 7, 8, 9, 10
    ],

    "SHAP Rank": [
        1, 2, 3, 5, 6, 4, 11, 7, 8, 9, 10
    ]
})


rank_cols = [
    "Stat Rank",
    "Built-in Rank",
    "Permutation Rank",
    "SHAP Rank"
]

rank_df["Average Rank"] = (
    rank_df[rank_cols]
    .mean(axis=1)
)

rank_df = (
    rank_df
    .sort_values("Average Rank")
    .reset_index(drop=True)
)


# ============================================================
# SHAP HELPER
# ============================================================

def get_local_shap(input_data):

    preprocessor = model.named_steps[
        "preprocess_before_smote"
    ]

    postprocessor = model.named_steps[
        "postprocess"
    ]

    tree_model = model.named_steps[
        "model"
    ]

    # Transform raw input
    X_pre = preprocessor.transform(
        input_data
    )

    X_processed = postprocessor.transform(
        X_pre
    )

    # Feature names after OHE
    ohe = postprocessor.named_transformers_[
        "cat"
    ]

    encoded_cat_names = (
        ohe.get_feature_names_out(
            categorical_cols
        )
    )

    feature_names = np.concatenate([
        numerical_cols,
        encoded_cat_names
    ])

    # SHAP
    explainer = shap.TreeExplainer(
        tree_model
    )

    shap_output = explainer(
        X_processed
    )

    # Binary classifier
    if shap_output.values.ndim == 3:
        shap_values = (
            shap_output.values[0, :, 1]
        )
    else:
        shap_values = (
            shap_output.values[0]
        )

    # Aggregate dummy variables
    # back to original features
    contributions = {}

    for feature in numerical_cols:

        idx = np.where(
            feature_names == feature
        )[0]

        contributions[feature] = (
            shap_values[idx].sum()
        )

    for feature in categorical_cols:

        idx = [
            i
            for i, name in enumerate(
                feature_names
            )
            if name.startswith(
                feature + "_"
            )
        ]

        contributions[feature] = (
            shap_values[idx].sum()
        )

    result = pd.DataFrame({
        "Feature": contributions.keys(),
        "SHAP Contribution":
            contributions.values()
    })

    result[
        "Absolute Contribution"
    ] = (
        result[
            "SHAP Contribution"
        ].abs()
    )

    result = (
        result
        .sort_values(
            "Absolute Contribution",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "💳 Credit Risk Project"
)

page = st.sidebar.radio(
    "เลือกหน้า",
    [
        "Prediction",
        "Model Performance",
        "Model Explainability"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Final model: Decision Tree (No SMOTENC)"
)

st.sidebar.warning(
    "Educational prototype only. "
    "Not intended for real lending decisions."
)


# ============================================================
# PAGE 1 : PREDICTION
# ============================================================

if page == "Prediction":

    st.title(
        "💳 Credit Risk Prediction"
    )

    st.write(
        """
        กรอกข้อมูลตัวอย่างผู้กู้
        เพื่อให้ Final Model จำแนกเป็น
        **Default** หรือ **Non-default**
        พร้อมอธิบายผลด้วย SHAP
        """
    )

    st.info(
        "Final Model: Decision Tree — No SMOTENC"
    )

    st.warning(
        "ต้นแบบนี้จัดทำเพื่อการศึกษาเท่านั้น "
        "ไม่ควรใช้เพื่ออนุมัติหรือปฏิเสธสินเชื่อจริง"
    )

    st.divider()

    st.subheader(
        "1. ข้อมูลผู้กู้"
    )

    left, right = st.columns(2)


    # ----------------------------
    # LEFT COLUMN
    # ----------------------------

    with left:

        person_age = st.number_input(
            "Age (อายุ)",
            min_value=18,
            max_value=100,
            value=26
        )

        person_income = st.number_input(
            "Annual Income (รายได้ต่อปี)",
            min_value=1,
            value=55000,
            step=1000
        )

        person_emp_length = st.number_input(
            "Employment Length (ปีที่ทำงาน)",
            min_value=0.0,
            max_value=60.0,
            value=4.0,
            step=1.0
        )

        person_home_ownership = st.selectbox(
            "Home Ownership",
            [
                "RENT",
                "MORTGAGE",
                "OWN",
                "OTHER"
            ]
        )

        cb_person_cred_hist_length = st.number_input(
            "Credit History Length (ปี)",
            min_value=0,
            max_value=50,
            value=4
        )


    # ----------------------------
    # RIGHT COLUMN
    # ----------------------------

    with right:

        loan_intent = st.selectbox(
            "Loan Intent",
            [
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "PERSONAL",
                "DEBTCONSOLIDATION",
                "HOMEIMPROVEMENT"
            ]
        )

        loan_grade = st.selectbox(
            "Loan Grade",
            [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G"
            ]
        )

        loan_amnt = st.number_input(
            "Loan Amount",
            min_value=1,
            value=8000,
            step=500
        )

        loan_int_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=40.0,
            value=10.99,
            step=0.1
        )

        cb_person_default_on_file = st.selectbox(
            "Previous Default on File",
            [
                "N",
                "Y"
            ]
        )


    # Automatically calculate
    # loan-to-income ratio

    loan_percent_income = (
        loan_amnt / person_income
        if person_income > 0
        else 0
    )

    st.metric(
        "Loan / Income Ratio",
        f"{loan_percent_income:.3f}"
    )

    st.caption(
        "ค่านี้คำนวณอัตโนมัติจาก "
        "Loan Amount ÷ Annual Income"
    )


    # ----------------------------
    # BUILD INPUT DATAFRAME
    # ----------------------------

    input_data = pd.DataFrame({
        "person_age": [
            person_age
        ],

        "person_income": [
            person_income
        ],

        "person_home_ownership": [
            person_home_ownership
        ],

        "person_emp_length": [
            person_emp_length
        ],

        "loan_intent": [
            loan_intent
        ],

        "loan_grade": [
            loan_grade
        ],

        "loan_amnt": [
            loan_amnt
        ],

        "loan_int_rate": [
            loan_int_rate
        ],

        "loan_percent_income": [
            loan_percent_income
        ],

        "cb_person_default_on_file": [
            cb_person_default_on_file
        ],

        "cb_person_cred_hist_length": [
            cb_person_cred_hist_length
        ]
    })


    st.divider()


    # ----------------------------
    # PREDICT BUTTON
    # ----------------------------

    if st.button(
        "🔍 Analyze Credit Risk",
        type="primary",
        use_container_width=True
    ):

        prediction = int(
            model.predict(
                input_data
            )[0]
        )

        score = float(
            model.predict_proba(
                input_data
            )[0, 1]
        )


        st.subheader(
            "2. Prediction Result"
        )

        c1, c2 = st.columns(2)


        with c1:

            if prediction == 1:

                st.error(
                    "🔴 Predicted Class: DEFAULT"
                )

            else:

                st.success(
                    "🟢 Predicted Class: NON-DEFAULT"
                )


        with c2:

            st.metric(
                "Default model score",
                f"{score:.3f}"
            )


        st.caption(
            "Default model score ไม่ควรตีความตรง ๆ "
            "ว่าเป็นโอกาส Default จริง "
            "เนื่องจาก Final Decision Tree "
            "ยังมีข้อจำกัดด้าน probability calibration"
        )


        # ----------------------------
        # SHAP
        # ----------------------------

        st.divider()

        st.subheader(
            "3. Why did the model predict this?"
        )

        shap_df = get_local_shap(
            input_data
        )

        top_shap = (
            shap_df
            .head(8)
            .copy()
        )

        display_shap = (
            top_shap[
                [
                    "Feature",
                    "SHAP Contribution"
                ]
            ]
            .copy()
        )

        display_shap[
            "Direction"
        ] = np.where(
            display_shap[
                "SHAP Contribution"
            ] > 0,
            "Toward Default",
            "Toward Non-default"
        )

        st.dataframe(
            display_shap,
            use_container_width=True,
            hide_index=True
        )


        # SHAP graph
        graph_df = (
            top_shap
            .sort_values(
                "SHAP Contribution"
            )
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        colors = [
            "tomato"
            if value > 0
            else "steelblue"

            for value in graph_df[
                "SHAP Contribution"
            ]
        ]

        ax.barh(
            graph_df["Feature"],
            graph_df[
                "SHAP Contribution"
            ],
            color=colors
        )

        ax.axvline(
            0,
            color="black",
            linewidth=1
        )

        ax.set_xlabel(
            "SHAP Contribution"
        )

        ax.set_title(
            "Local SHAP Explanation"
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        st.caption(
            "SHAP > 0 หมายถึงผลักการทำนายไปทาง Default "
            "ส่วน SHAP < 0 หมายถึงผลักไปทาง Non-default "
            "และไม่ได้หมายถึงความสัมพันธ์เชิงเหตุและผล"
        )


# ============================================================
# PAGE 2 : MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title(
        "📊 Model Performance"
    )

    st.write(
        """
        หน้านี้สรุปผลของ Final Model
        และเปรียบเทียบการทดลอง
        Logistic Regression,
        Decision Tree และ Random Forest
        ก่อนและหลังใช้ SMOTENC
        """
    )

    st.divider()

    st.subheader(
        "Final Test Performance"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Recall",
        f"{test_metrics['Recall']:.4f}"
    )

    c2.metric(
        "Precision",
        f"{test_metrics['Precision']:.4f}"
    )

    c3.metric(
        "F1",
        f"{test_metrics['F1-Score']:.4f}"
    )

    c4.metric(
        "Balanced Accuracy",
        f"{test_metrics['Balanced Accuracy']:.4f}"
    )

    c5.metric(
        "Accuracy",
        f"{test_metrics['Accuracy']:.4f}"
    )


    st.caption(
        "Final Model = Decision Tree without SMOTENC"
    )


    st.divider()

    st.subheader(
        "Cross Validation vs Test Set"
    )

    st.dataframe(
        cv_test_df.round(4),
        use_container_width=True,
        hide_index=True
    )


    st.write(
        """
        Test Set ให้ผลใกล้เคียงกับ
        Cross Validation มาก
        ซึ่งแสดงว่าประสิทธิภาพของโมเดล
        ไม่ได้ลดลงอย่างผิดปกติเมื่อเจอข้อมูลที่ไม่เคยเห็น
        """
    )


    st.divider()

    st.subheader(
        "6 Model Experiments"
    )

    st.dataframe(
        model_results.round(4),
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Effect of SMOTENC on Recall"
    )

    recall_comparison = pd.DataFrame({
        "No SMOTENC": [
            0.5553,
            0.7646,
            0.7196
        ],

        "SMOTENC": [
            0.7570,
            0.7716,
            0.7470
        ]
    }, index=[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ])

    st.bar_chart(
        recall_comparison
    )

    st.write(
        """
        **สรุป:** SMOTENC ช่วยเพิ่ม Recall
        ในทั้งสามโมเดล แต่ผลชัดที่สุดใน
        Logistic Regression

        อย่างไรก็ตาม Recall ที่สูงขึ้น
        มักมาพร้อมกับ Precision ที่ลดลง
        จึงต้องพิจารณา trade-off ร่วมกัน
        """
    )


    st.divider()

    st.subheader(
        "Final Model Parameters"
    )

    st.code(
        """
criterion = entropy
max_depth = None
min_samples_leaf = 1
min_samples_split = 2
        """
    )


    st.divider()

    st.subheader(
        "Probability Calibration"
    )

    brier_df = pd.DataFrame({
        "Brier Score": [
            0.1043,
            0.1709
        ]
    }, index=[
        "Final Model",
        "Baseline"
    ])

    st.bar_chart(
        brier_df
    )

    st.write(
        """
        Final Model มี Brier Score = **0.1043**
        ซึ่งต่ำกว่า Baseline = **0.1709**

        อย่างไรก็ตาม Calibration Curve
        พบว่า Decision Tree ให้ probability
        ค่อนข้างสุดโต่ง จึงไม่ควรตีความ
        model score เป็นความเสี่ยง Default จริงโดยตรง
        """
    )


# ============================================================
# PAGE 3 : MODEL EXPLAINABILITY
# ============================================================

elif page == "Model Explainability":

    st.title(
        "🔎 Model Explainability"
    )

    st.write(
        """
        หน้านี้อธิบายว่า Feature ใด
        มีบทบาทต่อการทำนายของ Final Model
        โดยใช้ SHAP และเปรียบเทียบ
        กับผล Statistical Analysis,
        Built-in Feature Importance
        และ Permutation Importance
        """
    )


    st.divider()

    st.subheader(
        "Global SHAP Importance"
    )

    shap_chart_df = (
        global_shap
        .set_index("Feature")
    )

    st.bar_chart(
        shap_chart_df
    )

    st.dataframe(
        global_shap.round(4),
        use_container_width=True,
        hide_index=True
    )


    st.success(
        "loan_percent_income "
        "เป็น Feature ที่มี Mean |SHAP| "
        "สูงที่สุดใน Final Model"
    )


    st.divider()

    st.subheader(
        "Cross-method Feature Ranking"
    )

    st.dataframe(
        rank_df.round(2),
        use_container_width=True,
        hide_index=True
    )


    st.write(
        """
        **วิธีอ่าน**

        - Rank 1 = สำคัญที่สุดในวิธีนั้น
        - Average Rank ต่ำ = มีแนวโน้มเด่นสม่ำเสมอหลายวิธี
        """ 
    )


    st.subheader(
        "Consensus Ranking"
    )

    consensus_chart = (
        rank_df[
            [
                "Feature",
                "Average Rank"
            ]
        ]
        .sort_values(
            "Average Rank",
            ascending=False
        )
        .set_index("Feature")
    )

    st.bar_chart(
        consensus_chart
    )


    st.divider()

    st.subheader(
        "Main Findings"
    )

    st.markdown(
        """
        **1. loan_percent_income**

        เด่นที่สุดอย่างสม่ำเสมอ:
        Statistical Rank = 2,
        Built-in = 1,
        Permutation = 1,
        SHAP = 1

        **2. loan_grade**

        มีความสำคัญสูงทั้งในเชิงสถิติ
        และ SHAP

        **3. loan_int_rate**

        เด่นมากใน Statistical Analysis
        แต่ความสำคัญลดลงเมื่อพิจารณา
        ร่วมกับ Feature อื่นในโมเดล

        **4. cb_person_default_on_file**

        มีความสัมพันธ์ทางสถิติ
        แต่ Final Model ใช้ Feature นี้
        ค่อนข้างน้อย

        สิ่งนี้แสดงว่า
        **Statistical association**
        และ **predictive importance**
        ไม่ใช่สิ่งเดียวกัน
        """
    )


    st.divider()

    st.info(
        """
        Feature Importance และ SHAP
        แสดงว่าตัวแปรมีบทบาทต่อการทำนายของโมเดล
        แต่ไม่สามารถใช้สรุปว่า Feature นั้น
        เป็นสาเหตุของ Default ได้
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Risk Analysis Prototype | "
    "Educational Data Science Project"
)
