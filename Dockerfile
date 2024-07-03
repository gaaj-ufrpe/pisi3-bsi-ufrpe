###############
# BUILD IMAGE #
###############
FROM python:3.11
#python
RUN python -m pip install --upgrade pip
COPY . .
#requirements
RUN pip install -r requirements.txt --upgrade 
#streamlit
EXPOSE 8501
ENTRYPOINT [ "streamlit", "run" ]
CMD ["Home.py"]
