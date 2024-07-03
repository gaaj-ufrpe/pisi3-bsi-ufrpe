###############
# BUILD IMAGE #
###############
FROM python:3.11
#python
RUN python -m pip install --upgrade pip
#app
WORKDIR /home/app
COPY data data
COPY pages pages
COPY reports reports
COPY requirements.txt .
COPY *.py ./
#requirements
RUN pip install -r requirements.txt --upgrade 
#streamlit
EXPOSE 8501
ENTRYPOINT [ "streamlit", "run" ]
CMD ["Home.py"]
