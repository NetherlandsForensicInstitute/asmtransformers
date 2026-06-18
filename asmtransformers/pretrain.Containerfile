FROM python:3.13
ARG PIP_PYPI_URL=https://pypi.org/simple
COPY --parents asmtransformers scripts pyproject.toml pdm.lock README.md /app
WORKDIR /app
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --break-system-packages --root-user-action ignore pdm
RUN pdm config pypi.verify_ssl False
RUN pdm config pypi.url $PIP_PYPI_URL
RUN pdm install --prod --no-editable
CMD ["bash"]
