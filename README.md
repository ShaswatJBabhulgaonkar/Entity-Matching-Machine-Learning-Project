# Entity Matching using Machine Learning

This GitHub repo contains project which is a part of my work sample that I recently worked on. It is an entity 
matching project where source and target data needs to be matched using machine learning.
It contains following sections: 
1) infra 
2) ml_pipeline
3) images
4) .circleci
5) Argo-workflows

## infra

The infra section consists of infrastructure code that I wrote using terraform. Terraform is used here as 
Infrastructure as Code(IaC) to build the necessary cloud infrastructure within AWS. Some of the AWS services
for which the infrastructure is built using terraform are s3, glue database, EKS, ECR. The modules section 
within infra contains code for terraform modules which is imported in dev to build necessary infrastructure.

## ml_pipeline

The ml_pipeline section contains ML codebase for entity matching. 

The first step in the pipeline is Importing which is performed in [import_data.py]

- generate run_id
- look up last target id
- import new tgts based on id
- add new records to tgt blocks as needed
- look up most recent src based on modified_at
- create specific blocks for these new records in this particular run
- look up most recent imported feedback based on modified_at
- import new feedback based on date range

The second step in the pipeline is Feature Engineering which is performed in [feature_eng.py]

- load in src, tgt
- preprocessing on src_df and tgt_df
- create pairs_df
- compute features
- write out features

The third step in the pipeline is Model Training which uses a helper class defined in [training.py] 
And then a [sci-kit learn pipeline object](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) 
defined in [model.py]

- lookup run info
- pull in most recent feedback. join features.
- load pre-trained model
- train
- save post-trained model

The fourth step in the pipeline is Model Inference which uses a helper class defined in [inference.py]
that loads up the most recently trained model of the appropriate type.

- load trained model by run_id and pretrain_model_id
- load all features by run_id.
- write out scores.
- Combine records, features, and scores 

The fifth and the last step in the pipeline is Exporting which is performed in [export.py] 
where we get the get the final output for the entity matched data.

## images

The images section consists of Dockerfile and Makefile.

The Dockerfile contains the require installations and dependencies.

The Makefile defines a set of commands for a docker image such as docker-build, docker-tag, docker-push.

## .circleci

The .circleci section consists of config.yml file.

CircleCI is used as a CI/CD tool which is configured using config.yml.

The CircleCI everytime checks for all the required basic checks and builds the docker image, tags it and pushes 
it to the ECR repository. The latest docker image with the tag which is deployed in ECR is used in production to
run jobs on EMR.

## Argo-workflows

The Argo-workflows section consists of workflow-template.yaml, cron-workflow.yaml, emr-job template.yaml and 
configuration-overrides.json files.

Argo is used here as a workflow orchestration tool for running jobs in production on EMR.

The workflow-template.yaml consists of a sequential workflow template for the entire ml_pipeline jobs such as import, feature engineering, training, modeling, inference and export. The cron-workflow.yaml is used to run jobs at a scheduled 
time frame which uses workflow-template to run jobs in production.