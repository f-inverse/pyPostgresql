# ==================================================================
# module list
# ------------------------------------------------------------------
# Ubuntu           20.04
# OpenMPI          latest       (apt)
# cmake            3.16.3       (apt)
# MKL              2020.4-912   (apt)
# arrayfire        3.7.3        (git, CPU backend)
# libsndfile       latest       (apt)
# oneDNN           v2.0         (git)
# Gloo             1da2117      (git)
# FFTW             latest       (apt)
# KenLM            0c4dd4e      (git)
# GLOG             latest       (apt)
# gflags           latest       (apt)
# python3          latest       (apt)
# ==================================================================

#############################################################################
#                             APT IMAGE + CMAKE                             #
#############################################################################

#FROM ubuntu:20.04 as cpu_base_builder
FROM postgres:latest

ENV APT_INSTALL="apt-get install -y --no-install-recommends"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        wget \
        git \
        g++ \
        postgresql \
        postgresql-contrib \
        cmake \
        sudo \
        python3-pip \
        python3-dev \
        libpq-dev && \
    apt-get clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install psycopg2

#############################################################################
#                        Configure Postgres Database                        #
#############################################################################

FROM library/postgres

ENV POSTGRES_USER docker
ENV POSTGRES_PASSWORD docker
ENV POSTGRES_DB docker

#psql -U docker

#CREATE TABLE mnist_fed (
#    id      SERIAL PRIMARY KEY,
#    image   INTEGER[][] NOT NULL, 
#    label   INT
#);