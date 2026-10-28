
from dotenv import load_dotenv
import streamlit as st

from typing import List, Optional
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# Load environment variables
load_dotenv()


# -------------------------------
# 1. Pydantic Model
# -------------------------------

class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str] = Field(default_factory=list)
    director: Optional[str] = None
    writer: Optional[str] = None
    main_cast: List[str] = Field(default_factory=list)
    rating: Optional[str] = None
    summary: str


# -------------------------------
# 2. Initialise Parser
# -------------------------------

parser = PydanticOutputParser(pydantic_object=Movie)


# -------------------------------
# 3. Initialise LLM
# -------------------------------

@st.cache_resource
def load_model():
    return init_chat_model(
        model="openai/gpt-oss-120b",
        model_provider="groq",
        temperature=0
    )


# -------------------------------
# 4. Create Prompt Template
# -------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a professional movie information assistant.

        Provide well-organised information about the requested movie.

        Requirements:
        - Include the correct movie title.
        - Provide the release year, genre, director, writer and main cast.
        - Include a rating when known.
        - Write a clear summary.
        - Do not invent facts.
        - Use null for unknown optional fields.
        - Use an empty list if a list field is unknown.

        {format_instructions}
        """
    ),
    (
        "human",
        "Provide information about this movie: {movie_name}"
    )
])


# -------------------------------
# 5. Streamlit Page Configuration
# -------------------------------

st.set_page_config(
    page_title="CineInfo AI",
    page_icon="🎬",
    layout="wide"
)


# -------------------------------
# 6. Custom Styling
# -------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 17px;
            color: #9CA3AF;
            margin-bottom: 30px;
        }

        .movie-card {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.3);
            margin-bottom: 15px;
        }

        .section-title {
            font-size: 22px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------
# 7. Header
# -------------------------------

st.markdown(
    '<div class="main-title">🎬 CineInfo AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover movie information using LangChain and Groq AI'
    '</div>',
    unsafe_allow_html=True
)


# -------------------------------
# 8. Movie Search Form
# -------------------------------

with st.form("movie_search_form"):

    movie_name = st.text_input(
        "Movie name",
        placeholder="e.g. Interstellar, Inception, The Dark Knight"
    )

    submitted = st.form_submit_button(
        "🔍 Discover Movie",
        use_container_width=True,
        type="primary"
    )


# -------------------------------
# 9. Generate Movie Information
# -------------------------------

if submitted:

    if not movie_name.strip():
        st.warning("Please enter a movie name.")

    else:
        try:
            with st.spinner(
                f"Getting information about {movie_name}..."
            ):

                model = load_model()

                final_prompt = prompt.invoke({
                    "movie_name": movie_name.strip(),
                    "format_instructions":
                        parser.get_format_instructions()
                })

                response = model.invoke(final_prompt)

                movie = parser.parse(response.content)

            # Save the result for subsequent Streamlit reruns
            st.session_state["movie_result"] = movie.model_dump()

        except Exception as e:
            st.error(
                "Unable to retrieve movie information. "
                "Please try again."
            )

            with st.expander("Technical details"):
                st.code(str(e))


# -------------------------------
# 10. Display Movie Information
# -------------------------------

if "movie_result" in st.session_state:

    data = st.session_state["movie_result"]

    st.divider()

    # Movie title
    st.title(f"🎥 {data['title']}")

    # Basic information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Release Year",
            data["release_year"]
            if data["release_year"] is not None
            else "Unknown"
        )

    with col2:
        st.metric(
            "Rating",
            data["rating"] or "Unknown"
        )

    with col3:
        st.metric(
            "Genres",
            len(data["genre"])
        )

    st.divider()

    # Movie details
    left_col, right_col = st.columns(2)

    with left_col:

        st.subheader("🎭 Movie Details")

        st.markdown(
            f"**Director:** {data['director'] or 'Unknown'}"
        )

        st.markdown(
            f"**Writer:** {data['writer'] or 'Unknown'}"
        )

        st.markdown("**Genres**")

        if data["genre"]:
            for genre in data["genre"]:
                st.markdown(f"- {genre}")
        else:
            st.write("Unknown")

    with right_col:

        st.subheader("🌟 Main Cast")

        if data["main_cast"]:
            for actor in data["main_cast"]:
                st.markdown(f"- {actor}")
        else:
            st.write("Unknown")

    st.divider()

    # Movie summary
    st.subheader("📖 Movie Summary")

    st.write(data["summary"])

    # Structured JSON output
    with st.expander("View Structured JSON"):

        st.json(data)

    # Download JSON
    st.download_button(
        label="⬇️ Download Movie Information",
        data=__import__("json").dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        file_name="movie_information.json",
        mime="application/json",
        use_container_width=True
    )


# -------------------------------
# 11. Footer
# -------------------------------

st.divider()

st.caption(
    "Built with Streamlit, LangChain, Pydantic and Groq. "
    "AI-generated information may contain errors."
)