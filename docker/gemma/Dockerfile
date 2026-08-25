FROM ollama/ollama:latest

ENV OLLAMA_MODEL=gemma4:e4b

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 11434

ENTRYPOINT ["/start.sh"]
