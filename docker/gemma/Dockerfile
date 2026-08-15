FROM ollama/ollama:latest

ENV OLLAMA_MODEL=gemma2:9b

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 11434

ENTRYPOINT ["/start.sh"]
