import streamlit as st
import joblib

from src.preprocessing import preprocess_text


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM STYLING
# ==========================================================

st.html("""
<style>

/* ==========================================================
   MAIN BACKGROUND
========================================================== */

.stApp {

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(0, 180, 255, 0.12),
            transparent 30%
        ),

        radial-gradient(
            circle at 90% 15%,
            rgba(120, 70, 255, 0.13),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #06111f 0%,
            #0a1730 45%,
            #111936 75%,
            #080f20 100%
        );

    color: #eaf4ff;
}


/* ==========================================================
   MAIN CONTENT
========================================================== */

.block-container {

    max-width: 1200px;

    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #081a2d 0%,
            #0b1428 100%
        );

    border-right:
        1px solid rgba(0, 200, 255, 0.15);
}


section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {

    color: #ffffff;
}


section[data-testid="stSidebar"] p {

    color: #a9bdd1;
}


/* ==========================================================
   TITLES
========================================================== */

h1 {

    color: #ffffff !important;

    font-weight: 800 !important;
}


h2,
h3 {

    color: #eaf4ff !important;
}


/* ==========================================================
   INFO BOX
========================================================== */

div[data-testid="stAlert"] {

    border-radius: 12px;

    border:
        1px solid rgba(0, 200, 255, 0.18);
}


/* ==========================================================
   TEXT AREA
========================================================== */

div[data-testid="stTextArea"] textarea {

    background-color:
        #091a2c !important;

    color:
        #edf7ff !important;

    border:
        1px solid #28445f !important;

    border-radius:
        14px !important;

    font-size:
        16px !important;

    line-height:
        1.6 !important;

    padding:
        15px !important;

    box-shadow:
        0 5px 20px rgba(0, 0, 0, 0.15) !important;
}


/* Placeholder */

div[data-testid="stTextArea"] textarea::placeholder {

    color:
        #7892aa !important;

    opacity:
        1 !important;
}


/* Text area focus */

div[data-testid="stTextArea"] textarea:focus {

    border:
        1px solid #00cfff !important;

    box-shadow:
        0 0 0 2px rgba(0, 207, 255, 0.10) !important;

    outline:
        none !important;
}


/* Text area label */

div[data-testid="stTextArea"] label {

    color:
        #dcecff !important;

    font-weight:
        600 !important;

    font-size:
        15px !important;
}


/* ==========================================================
   BUTTONS
========================================================== */

.stButton > button {

    background:
        #0c2136;

    color:
        #dcecff;

    border:
        1px solid rgba(0, 200, 255, 0.18);

    border-radius:
        10px;

    font-weight:
        600;

    transition:
        all 0.2s ease;
}


.stButton > button:hover {

    background:
        #12314d;

    color:
        #ffffff;

    border-color:
        #00cfff;

    transform:
        translateY(-1px);
}


/* ==========================================================
   PRIMARY CLASSIFY BUTTON
========================================================== */

.stButton > button[kind="primary"] {

    background:
        linear-gradient(
            90deg,
            #006eff,
            #00bfe9
        );

    color:
        white;

    border:
        none;

    font-size:
        16px;

    font-weight:
        700;

    min-height:
        48px;

    box-shadow:
        0 7px 22px rgba(0, 160, 255, 0.20);
}


.stButton > button[kind="primary"]:hover {

    background:
        linear-gradient(
            90deg,
            #0082ff,
            #14d4ff
        );

    box-shadow:
        0 9px 28px rgba(0, 180, 255, 0.30);
}


/* ==========================================================
   METRICS
========================================================== */

div[data-testid="stMetric"] {

    background:
        rgba(12, 30, 50, 0.75);

    border:
        1px solid rgba(0, 200, 255, 0.12);

    border-radius:
        12px;

    padding:
        12px;
}


div[data-testid="stMetricLabel"] {

    color:
        #8ea8bf !important;
}


div[data-testid="stMetricValue"] {

    color:
        #ffffff !important;
}


/* ==========================================================
   DIVIDERS
========================================================== */

hr {

    border-color:
        rgba(255, 255, 255, 0.08);
}


/* ==========================================================
   FOOTER
========================================================== */

.stCaption {

    color:
        #7189a1 !important;
}


/* ==========================================================
   HIDE STREAMLIT BRANDING
========================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""")


# ==========================================================
# STATE MANAGEMENT
# ==========================================================

if "news_text" not in st.session_state:

    st.session_state.news_text = ""


def set_example_text(text):

    st.session_state.news_text = text


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

@st.cache_resource
def load_models():

    model = joblib.load(
        "models/news_category_svm.pkl"
    )

    tfidf = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    class_names = joblib.load(
        "models/class_names.pkl"
    )

    return model, tfidf, class_names


try:

    model, tfidf, class_names = load_models()

except Exception:

    st.error(
        "⚠️ Models not found. "
        "Please ensure they exist in the 'models/' directory."
    )

    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("📌 About the Model")

    st.write(
        "**Dataset:** AG News\n\n"
        "**Feature Extraction:** TF-IDF\n\n"
        "**Classifier:** Tuned Linear SVM"
    )

    st.divider()

    st.write("**Categories:**")

    st.write(
        "🌍 World  |  "
        "⚽ Sports  |  "
        "💼 Business  |  "
        "💻 Sci/Tech"
    )

    st.divider()

    st.subheader("📊 Performance")

    col1, col2 = st.columns(2)

    col1.metric(
        "Accuracy",
        "92.36%"
    )

    col2.metric(
        "F1-Score",
        "92.34%"
    )


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "📰 News Category Classifier"
)

st.info(
    "Classify news articles into World, Sports, "
    "Business, or Technology using NLP and a Linear SVM."
)


# ==========================================================
# EXAMPLE NEWS SECTION
# ==========================================================

st.subheader("💡 Try an Example")


ex1, ex2, ex3, ex4 = st.columns(4)


ex_world = (
    "Government leaders from several countries met today "
    "to discuss international relations and a new agreement "
    "aimed at improving cooperation between the nations."
)


ex_sports = (
    "Manchester United secured a dramatic victory in the "
    "Premier League after scoring a late goal in the final "
    "minutes of the match."
)


ex_business = (
    "The stock market rose sharply today as investors "
    "reacted positively to strong corporate earnings and "
    "improved economic forecasts."
)


ex_tech = (
    "A technology company has introduced a new artificial "
    "intelligence system designed to improve smartphone "
    "performance and battery life."
)


with ex1:

    st.button(
        "🌍 World",
        on_click=set_example_text,
        args=(ex_world,),
        use_container_width=True
    )


with ex2:

    st.button(
        "⚽ Sports",
        on_click=set_example_text,
        args=(ex_sports,),
        use_container_width=True
    )


with ex3:

    st.button(
        "💼 Business",
        on_click=set_example_text,
        args=(ex_business,),
        use_container_width=True
    )


with ex4:

    st.button(
        "💻 Technology",
        on_click=set_example_text,
        args=(ex_tech,),
        use_container_width=True
    )


st.write("")


# ==========================================================
# NEWS INPUT
# ==========================================================

news_text = st.text_area(
    "📝 Paste your news article below:",

    value=st.session_state.news_text,

    height=200,

    placeholder=(
        "Example: The global economy shifted today "
        "as major central banks adjusted interest rates..."
    )
)


# ==========================================================
# PREDICTION LOGIC
# ==========================================================

if st.button(
    "🔍 Classify News",
    type="primary",
    use_container_width=True
):

    # ======================================================
    # INPUT VALIDATION
    # ======================================================

    if not news_text.strip():

        st.warning(
            "⚠️ Please enter a news article "
            "before classification."
        )


    elif len(news_text.split()) < 5:

        st.warning(
            "⚠️ Please enter a longer news article "
            "for better classification."
        )


    else:

        # ==================================================
        # MODEL PREDICTION
        # ==================================================

        with st.spinner(
            "Analyzing text..."
        ):

            # ----------------------------------------------
            # PREPROCESS TEXT
            # ----------------------------------------------

            cleaned_text = preprocess_text(
                news_text
            )


            # ----------------------------------------------
            # TF-IDF TRANSFORMATION
            # ----------------------------------------------

            text_vector = tfidf.transform(
                [cleaned_text]
            )


            # ----------------------------------------------
            # PREDICT CATEGORY
            # ----------------------------------------------

            prediction = model.predict(
                text_vector
            )[0]


            category = class_names[
                prediction
            ]


            # ----------------------------------------------
            # SVM DECISION SCORES
            # ----------------------------------------------

            decision_scores = model.decision_function(
                text_vector
            )[0]


        # ==================================================
        # PREDICTION RESULT
        # ==================================================

        st.divider()

        st.success(
            f"### Predicted Category: **{category}**",
            icon="🎯"
        )


        # ==================================================
        # CLASSIFICATION SCORES
        # ==================================================

        st.subheader(
            "📊 Classification Scores"
        )


        # Create category + score pairs
        score_data = list(
            zip(
                class_names,
                decision_scores
            )
        )


        # Sort highest score first
        score_data.sort(
            key=lambda item: item[1],
            reverse=True
        )


        # Find score range
        min_score = min(
            score for _, score in score_data
        )


        max_score = max(
            score for _, score in score_data
        )


        score_range = (
            max_score - min_score
        )


        if score_range == 0:

            score_range = 1


        # ==================================================
        # SCORE CARDS
        # ==================================================

        for category_name, score in score_data:

            # ----------------------------------------------
            # Visual bar width
            # ----------------------------------------------

            visual_percentage = (
                (score - min_score)
                / score_range
            ) * 100


            # ----------------------------------------------
            # Highlight predicted category
            # ----------------------------------------------

            if category_name == category:

                border_color = "#00d4ff"

                background_color = "#102b42"

                icon = "🎯"

            else:

                border_color = "#24384d"

                background_color = "#0b1929"

                icon = "•"


            # ----------------------------------------------
            # HTML SCORE CARD
            #
            # IMPORTANT:
            # We use st.html(), NOT st.markdown().
            # This prevents the HTML from appearing
            # as plain text.
            # ----------------------------------------------

            st.html(
                f"""
                <div style="
                    background:{background_color};
                    border:1px solid {border_color};
                    border-radius:12px;
                    padding:14px 18px;
                    margin-bottom:10px;
                ">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                        margin-bottom:8px;
                    ">

                        <span style="
                            color:#eaf4ff;
                            font-weight:600;
                            font-size:15px;
                        ">
                            {icon} {category_name}
                        </span>


                        <span style="
                            color:#9fb6cc;
                            font-size:14px;
                        ">
                            {score:.4f}
                        </span>

                    </div>


                    <div style="
                        width:100%;
                        height:6px;
                        background:#182c40;
                        border-radius:10px;
                        overflow:hidden;
                    ">

                        <div style="
                            width:{visual_percentage:.1f}%;
                            height:100%;
                            background:
                                linear-gradient(
                                    90deg,
                                    #0077ff,
                                    #00d4ff
                                );
                            border-radius:10px;
                        ">
                        </div>

                    </div>

                </div>
                """
            )


        # ==================================================
        # SCORE EXPLANATION
        # ==================================================

        st.caption(
            "SVM decision scores show how strongly the "
            "model favors each category. They are not "
            "probability percentages."
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "News Category Classification | "
    "TF-IDF + Tuned Linear SVM | AG News Dataset"
)