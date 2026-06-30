-- Banco apontado para o checkpointer do LangGraph (implementação futura).
-- O banco da aplicação (`app`) é criado pela env POSTGRES_DB.
-- Este script roda apenas no primeiro boot, com o volume de dados vazio.
CREATE DATABASE checkpointer;
