FROM 711395599931.dkr.ecr.us-east-2.amazonaws.com/spark/emr-6.3.0:20210429
# comes with Python 3.7.9 as `python3`

USER root

# Pre gcc and unixODBC-devel for pyodbc
# see https://stackoverflow.com/questions/62387396/unable-install-sqlserver-tool-in-ubuntu-20/63187951#63187951

# get gcc through amazon "Development Tools"
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/compile-software.html
RUN yum groupinstall -y "Development Tools"
RUN yum install -y \
  unixODBC-devel \
  python3-devel

# MSSQL install
# see https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver15#redhat17
ENV ACCEPT_EULA=Y
RUN curl https://packages.microsoft.com/config/rhel/7/prod.repo > /etc/yum.repos.d/mssql-release.repo
RUN yum install -y \
  msodbcsql17

# Update pip prior to python installs
RUN python3 -m pip install --upgrade pip

# python dependencies
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# TODO have layered images for development and production
# install development dependencies
RUN yum install -y \
  vim \
  jq

# python development dependencies
ADD requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install -r /tmp/requirements-dev.txt

EXPOSE 8888

RUN mkdir /notebooks
WORKDIR /notebooks

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH /app
ENV LOGLEVEL=DEBUG

ENTRYPOINT ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--no-browser"]
