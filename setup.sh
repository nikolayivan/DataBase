mkdir -p ~/.streamlit/

echo "\
[global]\n\
dataFrameSerialization = "legacy"\n\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
