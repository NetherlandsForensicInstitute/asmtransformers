FROM python:3.13

COPY --parents asmtransformers scripts pyproject.toml pdm.lock README.md /app
WORKDIR /app
RUN pip install --cert "${SSL_CERT_FILE}" --trusted-host pypi.org --trusted-host files.pythonhosted.org --break-system-packages --root-user-action ignore pdm
RUN pdm config pypi.verify_ssl False
RUN pdm install
CMD ["bash"]