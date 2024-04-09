# AML Pipelines combined with GitHub Actions 

The ```.github/workflow/sweep-pipeline-job.yml``` file defines the following steps :
- Checking out the dev branch 
- Installing 3.11 Python
- RUnning some unit tests
- Log in into Azure
- Getting relevant Azure Resource Group and Workspace
- RUn AML Pipeline
  
For now we run the Naive Bayes pipeline which is defined in the code/aml/naive-bayes/src/pipelines folder. 

It has the following steps:
- data preprocessing
- hyperparameter sweep
- predictions step
- scoring step

To get more info about YAML pipelines schema, please check the following resources : 
- [Pipelines examples](https://github.com/Azure/azureml-examples/tree/main/cli/jobs/pipelines-with-components)
- [Azure CLI Pipeline YAML Reference](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-pipeline?view=azureml-api-2#examples)