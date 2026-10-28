import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()


class Movie(BaseModel):
    title:str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    writer: Optional[str]
    main_cast:List(str)    
    rating: Optional[str]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)
# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CineScope AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #f5f5f5;
    }

    .main-title {
        font-size: 44px;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #a5adbd;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .section-title {
        color: #ff4b6e;
        font-size: 23px;
        font-weight: 700;
        margin-top: 20px;
    }

    .info-card {
        background-color: #151c2c;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #283248;
        margin-bottom: 15px;
    }

    .stButton > button {
        background-color: #ff4b6e;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        min-height: 45px;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #e6385b;
        color: white;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LANGCHAIN MODEL
# --------------------------------------------------

@st.cache_resource
def get_model():
    return init_chat_model(
        model="openai/gpt-oss-120b",
        model_provider="groq",
        temperature=0.3
    )


# --------------------------------------------------
# MOVIE PROMPT
# --------------------------------------------------

template = """
You are CineScope AI, a professional movie information assistant.

Movie Title: {movie_name}

Provide information using the following format.

## 1. Basic Information

- **Title:**
- **Release Year:**
- **Genre:**
- **Director:**
- **Writer:**
- **Main Cast:**
- **Original Language:**
- **Country:**
- **Runtime:**
- **IMDb Rating:**

## 2. Detailed Story Summary

Write one paragraph of 80-120 words explaining the story,
main character, central conflict, and important developments.

Do not reveal the ending unless the user requests spoilers.

## 3. Quick Summary

Summarise the movie in 2-3 sentences.

## 4. Main Characters

List 3-5 important characters and their roles.

## 5. Themes and Highlights

Explain the main themes and what makes the movie special.

## 6. Final Verdict

Give a brief, balanced opinion about the movie.

RULES:
- Use clear, natural English.
- Never invent facts, cast members, or ratings.
- Mark uncertain information as unverified.
- Keep the sections concise and informative.
- Use Markdown headings and bullet points.
- Do not repeat information unnecessarily.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template)
])


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎬 CineScope AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover movies through AI-powered insights, '
    'stories, characters and cinematic details.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🎥 CineScope AI")

    st.write("Your personal movie information assistant.")

    st.divider()

    st.markdown("### ✨ Features")

    st.write("🎞️ Movie information")
    st.write("📖 Detailed story summaries")
    st.write("⚡ Quick summaries")
    st.write("👥 Character information")
    st.write("💡 Themes and highlights")

    st.divider()

    st.caption("Powered by LangChain + Groq")


# --------------------------------------------------
# MOVIE SEARCH
# --------------------------------------------------

st.markdown("### 🔍 Explore a Movie")

with st.form("movie_search_form"):
    movie_name = st.text_input(
        "Movie title",
        placeholder="Try Interstellar, Inception, or The Dark Knight"
    )

    submitted = st.form_submit_button(
        "✨ Generate Movie Insights"
    )


# --------------------------------------------------
# GENERATE MOVIE INFORMATION
# --------------------------------------------------

if submitted:

    if not movie_name.strip():
        st.warning("Please enter a movie title.")

    else:
        try:
            with st.spinner(
                f"🎬 Analysing {movie_name.strip()}..."
            ):
                model = get_model()

                chain = prompt | model

                response = chain.invoke({
                    "movie_name": movie_name.strip()
                })

                result = response.content

                if isinstance(result, list):
                    result = "\n".join(
                        str(item.get("text", item))
                        if isinstance(item, dict)
                        else str(item)
                        for item in result
                    )

                st.session_state["movie_result"] = result
                st.session_state["movie_name"] = movie_name.strip()

        except Exception as error:
            st.error(
                "Unable to generate movie information. "
                "Please check your API key, model configuration, "
                "and internet connection."
            )

            with st.expander("Technical error"):
                st.code(str(error))


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if st.session_state.get("movie_result"):

    result = st.session_state["movie_result"]
    searched_movie = st.session_state["movie_name"]

    st.divider()

    st.markdown(
        f"## 🎞️ {searched_movie}"
    )

    st.caption("AI-generated movie insights")

    # Separate the quick summary from the full response
    sections = result.split("## 3. Quick Summary")

    if len(sections) > 1:
        main_information = sections[0]

        remaining = sections[1].split("## 4. Main Characters")

        quick_summary = remaining[0].strip()

        st.markdown("### ⚡ Quick Summary")

        st.markdown(
            f"""
            <div class="info-card">
                {quick_summary.replace("<", "&lt;").replace(">", "&gt;")}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 📚 Complete Movie Information")

        st.markdown(main_information)

        st.markdown("## 👥 Main Characters and More")

        if len(remaining) > 1:
            st.markdown(
                "## 4. Main Characters" + remaining[1]
            )

    else:
        st.markdown(result)

    st.divider()

    st.caption(
        "Note: AI-generated details may contain errors. "
        "Verify ratings and factual information using reliable sources."
    )