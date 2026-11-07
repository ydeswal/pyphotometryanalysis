# Running and publishing the app

## On your own computer (recommended for long or many recordings)

Double-click `run_local.command` (Mac), `run_local.bat` (Windows) or run
`bash run_local.sh` (Linux). See the README section "Run it on your own
computer". There is no memory cap locally, and uploads up to 4 GB are allowed.

## The hosted site (Streamlit Community Cloud)

The site redeploys automatically whenever `main` on GitHub changes. Limits
worth knowing:

- **2.7 GB of memory** for the whole app, shared by everyone using it at that
  moment. One 48 h, 130 Hz recording now peaks around 1 GB during analysis. Two
  people analysing large files at once can still hit the limit. Use the local
  launcher for those.
- **Uploads up to 1 GB per file** (set in `.streamlit/config.toml`, which
  must be in a folder called `.streamlit` with the dot).
- If the page shows **"Oh no. Error running app"**: open
  <https://share.streamlit.io>, then **⋮ → Reboot app**. Use **⋮ → Logs** to
  see why it stopped.

## The "Ask the data" AI chat: adding free keys

The chat needs at least one free API key. Keys are never put in the code.

1. **Google Gemini (main, free):** sign in at <https://aistudio.google.com/apikey>
   and click *Create API key*. No card needed. Its free tier is about 1,500
   requests a day and 10 a minute per key (your key's exact limits are on the
   AI Studio rate-limit page).
2. **Groq (backup, free, recommended):** <https://console.groq.com/keys> →
   *Create API Key*. 1,000 requests a day for gpt-oss-120b. Used only when
   every Gemini model is busy or out of quota. Google's free models are
   sometimes overloaded ("high demand", HTTP 503); a Groq key keeps the chat
   working then.

   With the Gemini key alone the chat already tries three Gemini models in
   turn (gemini-3.8-flash → gemini-3.6-flash → gemini-3.5-flash-lite), each with
   its own quota, and waits and retries a busy model twice before moving on.
3. On <https://share.streamlit.io> open the app → **⋮ → Settings → Secrets**
   and paste:

       GEMINI_API_KEY = "your-gemini-key"
       GROQ_API_KEY = "your-groq-key"
       # optional: anyone opening the site must type this to use the chat
       CHAT_PASSWORD = "choose-one"

   Save; the app restarts with the chat ready. Keep keys private: never
   share a screenshot of this box. If one is seen, delete that key in AI
   Studio (or Groq) and make a new one. Without secrets, each person
   can paste their own key in the chat's **⚙ AI settings** (kept only for
   their browser session).

Optional settings in the same Secrets box:

    GEMINI_MODEL = "gemini-3.8-flash"     # if Google renames the model
    GEMINI_FALLBACK_MODELS = "gemini-3.6-flash, gemini-3.5-flash-lite"
    GROQ_MODEL = "openai/gpt-oss-120b"
    AI_BASE_URL = "https://openrouter.ai/api/v1"   # any OpenAI-compatible
    AI_MODEL = "some/model:free"                    # service, tried last
    AI_API_KEY = "key"

Running locally, put the same lines in `.streamlit/secrets.toml` (already in
`.gitignore`, so it is never committed).

Privacy: on Gemini's free tier Google may use what is sent to improve its
products, and human reviewers may read it. The chat sends computed summaries
(numbers, times, settings), never whole files. For data that must stay
confidential, use a paid key or keep the chat switched off.

## Publishing changes

    git add -A
    git commit -m "describe the change"
    git push

Work on a branch first if you want to check a change before the live site
picks it up. The site follows whichever branch it was deployed from.
