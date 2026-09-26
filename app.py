import os
import json
import streamlit as st
from dotenv import load_dotenv
from apify_client import ApifyClient
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================
# FAISS VECTOR DATABASE
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)
if not APIFY_TOKEN:
    raise ValueError("APIFY_TOKEN not found in .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")


# ============================================================
# CLIENTS
# ============================================================

apify_client = ApifyClient(APIFY_TOKEN)

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="eChai Startup Demo Day",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "profile_data" not in st.session_state:
    st.session_state.profile_data = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "show_full_profile" not in st.session_state:
    st.session_state.show_full_profile = False


# ============================================================
# CLEANING CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# TOP-LEVEL FIELDS TO REMOVE
# ------------------------------------------------------------

REMOVE_TOP_LEVEL = {
    "id",
    "publicIdentifier",
    "emails",
    "hiring",
    "premium",
    "influencer",
    "memorialized",
    "creator",
    "objectUrn",
    "registeredAt",
    "connectionsCount",
    "followerCount",
    "verified",
    "profileTopEducation",
    "profileActions",
    "profilePicture",
    "coverPicture",
    "photo",
    "profileLocales",
    "primaryLocale",
    "multiLocaleHeadline",
    "services",
    "featured",
    "composeOptionType",
    "moreProfiles",
    "sectionTotals",
    "originalQuery"
}


# ------------------------------------------------------------
# NESTED FIELDS TO REMOVE
# ------------------------------------------------------------

REMOVE_EXPERIENCE_FIELDS = {
    "companyUniversalName",
    "companyLinkedinUrl",
    "companyId",
    "companyLogo",
    "experienceGroupId"
}

REMOVE_EDUCATION_FIELDS = {
    "schoolLinkedinUrl",
    "schoolId",
    "schoolLogo"
}

REMOVE_CERTIFICATION_FIELDS = {
    "issuedByLink",
    "issuedByLogo"
}

REMOVE_SKILL_FIELDS = {
    "positions",
    "endorsements"
}


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def clean_date(value):
    """
    Converts:

    {
        "month": "Jul",
        "year": 2024,
        "text": "Jul 2024"
    }

    into:

    "Jul 2024"
    """

    if not isinstance(value, dict):
        return value

    # Prefer formatted date if available
    if value.get("text"):
        return value["text"]

    # Fallback
    month = value.get("month")
    year = value.get("year")

    if month and year:
        return f"{month} {year}"

    return value


def clean_location(value):
    """
    Converts LinkedIn's nested location object
    into a simple string.
    """

    if not isinstance(value, dict):
        return value

    parsed = value.get("parsed")

    if isinstance(parsed, dict):
        if parsed.get("text"):
            return parsed["text"]

    if value.get("linkedinText"):
        return value["linkedinText"]

    return value


def clean_current_position(position):
    """
    Remove irrelevant metadata from current position.
    """

    if not isinstance(position, dict):
        return position

    cleaned = {}

    for key, value in position.items():

        if key in REMOVE_EXPERIENCE_FIELDS:
            continue

        if key in {"startDate", "endDate"}:
            cleaned[key] = clean_date(value)

        else:
            cleaned[key] = value

    return cleaned


def clean_experience(experience):
    """
    Clean the complete experience list.
    """

    if not isinstance(experience, list):
        return experience

    cleaned_experience = []

    for exp in experience:

        if not isinstance(exp, dict):
            cleaned_experience.append(exp)
            continue

        cleaned = {}

        for key, value in exp.items():

            if key in REMOVE_EXPERIENCE_FIELDS:
                continue

            if key in {"startDate", "endDate"}:
                cleaned[key] = clean_date(value)

            else:
                cleaned[key] = value

        cleaned_experience.append(cleaned)

    return cleaned_experience


def clean_education(education):
    """
    Clean the complete education list.
    """

    if not isinstance(education, list):
        return education

    cleaned_education = []

    for edu in education:

        if not isinstance(edu, dict):
            cleaned_education.append(edu)
            continue

        cleaned = {}

        for key, value in edu.items():

            if key in REMOVE_EDUCATION_FIELDS:
                continue

            if key in {"startDate", "endDate"}:
                cleaned[key] = clean_date(value)

            else:
                cleaned[key] = value

        cleaned_education.append(cleaned)

    return cleaned_education


def clean_certifications(certifications):
    """
    Clean certification information.
    """

    if not isinstance(certifications, list):
        return certifications

    cleaned_certifications = []

    for cert in certifications:

        if not isinstance(cert, dict):
            cleaned_certifications.append(cert)
            continue

        cleaned = {}

        for key, value in cert.items():

            if key in REMOVE_CERTIFICATION_FIELDS:
                continue

            cleaned[key] = value

        cleaned_certifications.append(cleaned)

    return cleaned_certifications


def clean_skills(skills):
    """
    Remove LinkedIn-specific skill metadata such as
    positions and endorsements.
    """

    if not isinstance(skills, list):
        return skills

    cleaned_skills = []

    for skill in skills:

        if not isinstance(skill, dict):
            cleaned_skills.append(skill)
            continue

        cleaned = {}

        for key, value in skill.items():

            if key in REMOVE_SKILL_FIELDS:
                continue

            cleaned[key] = value

        cleaned_skills.append(cleaned)

    return cleaned_skills


# ============================================================
# MAIN PROFILE CLEANER
# ============================================================

def clean_linkedin_profile(data):
    """
    Takes the raw JSON returned by Apify and removes
    irrelevant LinkedIn/API/UI metadata.
    """

    if not isinstance(data, dict):
        raise ValueError(
            "LinkedIn profile must be a dictionary."
        )

    cleaned = {}

    # --------------------------------------------------------
    # Remove irrelevant top-level fields
    # --------------------------------------------------------

    for key, value in data.items():

        if key in REMOVE_TOP_LEVEL:
            continue

        cleaned[key] = value

    # --------------------------------------------------------
    # Clean location
    # --------------------------------------------------------

    if "location" in cleaned:

        cleaned["location"] = clean_location(
            cleaned["location"]
        )

    # --------------------------------------------------------
    # Clean current position
    # --------------------------------------------------------

    if "currentPosition" in cleaned:

        if isinstance(
            cleaned["currentPosition"],
            list
        ):

            cleaned["currentPosition"] = [
                clean_current_position(position)
                for position in cleaned["currentPosition"]
            ]

        elif isinstance(
            cleaned["currentPosition"],
            dict
        ):

            cleaned["currentPosition"] = (
                clean_current_position(
                    cleaned["currentPosition"]
                )
            )

    # --------------------------------------------------------
    # Clean experience
    # --------------------------------------------------------

    if "experience" in cleaned:

        cleaned["experience"] = clean_experience(
            cleaned["experience"]
        )

    # --------------------------------------------------------
    # Clean education
    # --------------------------------------------------------

    if "education" in cleaned:

        cleaned["education"] = clean_education(
            cleaned["education"]
        )

    # --------------------------------------------------------
    # Clean certifications
    # --------------------------------------------------------

    if "certifications" in cleaned:

        cleaned["certifications"] = (
            clean_certifications(
                cleaned["certifications"]
            )
        )

    # --------------------------------------------------------
    # Clean skills
    # --------------------------------------------------------

    if "skills" in cleaned:

        cleaned["skills"] = clean_skills(
            cleaned["skills"]
        )

    return cleaned


# ============================================================
# LINKEDIN PROFILE EXTRACTION
# ============================================================

def extract_linkedin_profile(linkedin_url):

    input_data = {
        "profileScraperMode": "Profile details no email ($4 per 1k)",
        "queries": [
            linkedin_url
        ],
    }

    # --------------------------------------------------------
    # Run Apify scraper
    # --------------------------------------------------------

    run = apify_client.actor(
        "harvestapi/linkedin-profile-scraper"
    ).call(
        run_input=input_data
    )

    # --------------------------------------------------------
    # Get dataset
    # --------------------------------------------------------

    dataset = apify_client.dataset(
        run.default_dataset_id
    )

    items = list(dataset.iterate_items())

    if not items:
        raise ValueError(
            "No profile data was returned by Apify."
        )

    # --------------------------------------------------------
    # Raw profile returned by Apify
    # --------------------------------------------------------

    raw_profile = items[0]

    # --------------------------------------------------------
    # CLEAN PROFILE
    # --------------------------------------------------------

    cleaned_profile = clean_linkedin_profile(
        raw_profile
    )

    return cleaned_profile


# ============================================================
# PROFILE SUMMARY
# ============================================================

def summarize_profile(profile_data):

    # --------------------------------------------------------
    # Convert cleaned profile into JSON text
    # --------------------------------------------------------

    profile_json = json.dumps(
        profile_data,
        indent=2,
        ensure_ascii=False
    )

    system_prompt = """
You are a professional networking assistant.

Your job is to analyze a LinkedIn profile and create a concise
networking-oriented summary.

Use ONLY information explicitly present in the supplied profile.

Do NOT invent:

- skills
- experience
- interests
- goals
- companies
- achievements
- intentions

If something is not available, say "Not specified".

Focus on information that would help another person decide:

"Why might I want to talk to this person at a networking event?"

Return the following sections:

## Professional Summary

A concise 3-5 sentence overview.

## Current Role

Current position and company.

## Expertise

Important technical/professional skills.

## Experience

Important previous roles and relevant background.

## Education

Relevant educational background.

## Interests

Professional or personal interests explicitly mentioned.

## What They Can Help With

Potential areas where this person could help others,
based only on their profile.

## What They May Be Interested In

Potential networking topics based only on explicit information.

## Conversation Starters

3-5 natural topics that someone could use to start a conversation.

## Networking Keywords

A list of 8-15 useful keywords.

Keep the summary concise and useful for a networking event.
"""

    user_prompt = f"""
Analyze this LinkedIn profile:

{profile_json}
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=1500
    )

    return response.choices[0].message.content


# ============================================================
# FIND RELEVANT NETWORKING OPPORTUNITIES
# ============================================================

def find_relevant_chunks(summary, k=20):

    results = vectorstore.similarity_search(
        summary,
        k=k
    )

    return results
# ============================================================
# FRONTEND
# ============================================================

st.title("Who should I talk to?")

st.markdown(
    "Find the people at this event you should meet."
)



# ============================================================
# INPUT
# ============================================================

linkedin_url = st.text_input(
    "LinkedIn Profile URL",
    placeholder="https://www.linkedin.com/in/example/"
)


# ============================================================
# EXTRACT BUTTON
# ============================================================

if st.button(
    "Extract my info.",
    type="primary"
):

    if not linkedin_url.strip():

        st.warning(
            "Please enter a LinkedIn profile URL."
        )

    elif "linkedin.com/in/" not in linkedin_url:

        st.error(
            "Please enter a valid LinkedIn profile URL."
        )

    else:

        try:

            with st.spinner(
                "Extracting LinkedIn profile..."
            ):

                profile = extract_linkedin_profile(
                    linkedin_url
                )

            # Save CLEANED profile
            st.session_state.profile_data = profile

            # Clear old summary
            st.session_state.summary = None

            # Reset full profile view
            st.session_state.show_full_profile = False

            st.success(
                "LinkedIn profile extracted successfully!"
            )

        except Exception as e:

            st.error(
                f"Error extracting profile: {str(e)}"
            )


# ============================================================
# SHOW PROFILE OPTIONS
# ============================================================

if st.session_state.profile_data:

    profile = st.session_state.profile_data

    st.divider()

    # --------------------------------------------------------
    # BASIC PROFILE INFORMATION
    # --------------------------------------------------------

    st.subheader("Profile")

    first_name = profile.get(
        "firstName",
        ""
    )

    last_name = profile.get(
        "lastName",
        ""
    )

    name = f"{first_name} {last_name}".strip()

    headline = profile.get(
        "headline",
        "Not specified"
    )

    linkedin = profile.get(
        "linkedinUrl",
        linkedin_url
    )

    st.markdown(
        f"### {name}"
    )

    st.write(headline)

    st.markdown(
        f"**LinkedIn:** {linkedin}"
    )

    # --------------------------------------------------------
    # BUTTONS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    # ========================================================
    # FULL PROFILE
    # ========================================================

    with col1:

        if st.button(
            "View Extracted Json",
            use_container_width=True
        ):

            st.session_state.show_full_profile = True

    # ========================================================
    # SUMMARY
    # ========================================================

    with col2:

        if st.button(
            "Summarize my LinkedIn",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Analyzing profile with Groq..."
                ):

                    summary = summarize_profile(
                        profile
                    )

                st.session_state.summary = summary

            except Exception as e:

                st.error(
                    f"Error generating summary: {str(e)}"
                )


# ============================================================
# FULL CLEANED PROFILE
# ============================================================

if (
    st.session_state.profile_data
    and st.session_state.get(
        "show_full_profile",
        False
    )
):

    st.divider()

    st.subheader(
        "Cleaned Extracted Profile"
    )

    st.json(
        st.session_state.profile_data
    )


# ============================================================
# SUMMARY
# ============================================================

if st.session_state.summary:

    st.divider()

    st.subheader(
        "My Summary based on LinkedIn" 
    )

    st.markdown(
        st.session_state.summary
    )

    st.divider()

    if st.button(
        "Find My Matches",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Finding relevant people..."
        ):

            results = find_relevant_chunks(
                st.session_state.summary,
                k=20
            )
        unique_results = []
        seen_sources = set()

        for doc in results:

            source = doc.metadata.get("source")

            if source not in seen_sources:

                seen_sources.add(source)
                unique_results.append(doc)

            if len(unique_results) == 5:
                break 
        st.subheader(
            "People you should should talk to"
        )

        for i, doc in enumerate(unique_results, start=1):

            st.markdown(
                f"### #{i} — {doc.metadata.get('name', 'Unknown')}"
            )

            st.write(
                f"**Type:** {doc.metadata.get('type', 'Unknown')}"
            )

            st.write(
                f"**URL:** {doc.metadata.get('url', 'Unknown')}"
            )

            st.write(doc.page_content)

            st.divider()
