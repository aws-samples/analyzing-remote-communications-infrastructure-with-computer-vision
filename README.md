# Analyzing Remote Communications Infrastructure with Computer Vision

This solution shows how to leverage ML tools to analyze on-the-ground or drone-based photographs. Images uploaded to an Amazon Simple Storage Service (Amazon S3) bucket are automatically analyzed by a customizable number of computer vision models. The results are inserted into an Amazon DynamoDB table and used to label the image with bounding boxes, confidence scores, and label name. Finally, a labeled version of the image is uploaded to the S3 bucket.

![Alt](./img/ArchitectureImage.png "Architecture Diagram")

## Try it yourself

To use this solution, an AWS account with access and permissions to deploy the following services is necessary:

- [SageMaker](https://aws.amazon.com/sagemaker/)
- [Lambda](https://aws.amazon.com/lambda/)
- [Amazon S3](https://aws.amazon.com/pm/serv-s3/?trk=fecf68c9-3874-4ae2-a7ed-72b6d19c8034&sc_channel=ps&s_kwcid=AL!4422!3!536452728638!e!!g!!amazon%20s3&ef_id=Cj0KCQjwtsCgBhDEARIsAE7RYh25D528BtMo_7MQlSCwtPFC4DtwRLsz9GenfDC5MDUzm-7uTo8WhCoaAvZREALw_wcB:G:s&s_kwcid=AL!4422!3!536452728638!e!!g!!amazon%20s3)
- [Amazon SQS](https://aws.amazon.com/pm/sqs/?ef_id=Cj0KCQjwtsCgBhDEARIsAE7RYh0IbDqCZT4jCFP-Idwaw4ti8WokbNIXkEoD4DN-MjZopamKrGrUpvsaAgXLEALw_wcB:G:s&s_kwcid=AL!4422!3!629393325349!!!g!!)
- [AWS Step Functions](https://aws.amazon.com/pm/step-functions/?ef_id=Cj0KCQjwtsCgBhDEARIsAE7RYh0zwE36wNL1-fH5Xyq654Ry5WgnNLDdBiOENFE6GSstODD14hReP6UaArV8EALw_wcB:G:s&s_kwcid=AL!4422!3!629393325319!!!g!!)
- [DynamoDB](https://aws.amazon.com/dynamodb/?trk=94bf4df1-96e1-4046-a020-b07a2be0d712&sc_channel=ps&s_kwcid=AL!4422!3!610000101513!e!!g!!dynamodb&ef_id=Cj0KCQjwtsCgBhDEARIsAE7RYh36RyGA0XaORfnQZ3YEeWHeX6slwdXwLAPSk1hy3qBVuuKjI_MZwAEaAkMtEALw_wcB:G:s&s_kwcid=AL!4422!3!610000101513!e!!g!!dynamodb)
- [Systems Manager](https://aws.amazon.com/systems-manager/?ef_id=Cj0KCQjwtsCgBhDEARIsAE7RYh3HfUyh6jufx52PHvaV69GHdpMnGky_OQYQ0gebq7IJmIrMYTtR7BUaAkxAEALw_wcB:G:s&s_kwcid=AL!4422!3!629393326000!!!g!!) (Parameter Store)
- [AWS CloudFormation](https://aws.amazon.com/cloudformation/)

### Before you begin

We provide all the necessary code for the functions and layers, and a CloudFormation template that deploys all the required resources for you. This architecture is built around an existing, running endpoint model, so we provide the instructions and resources to launch a sample one before deploying the CloudFormation stack.

If you already have available endpoints, then you may skip ahead to the CloudFormation step. However, since different models differ in their expected inputs and outputs, remember that you probably need to change the code in the functions to fit the input format and parameters of your own endpoints.

### Prepare a bucket with all the provided files

Before deploying the resources, we upload all the source files to an S3 bucket so it is easy to find. You must give it a unique name, but throughout the instructions we refer to this bucket as our resource bucket.

You should have the following files:

- `template.yaml`: the CloudFormation template
- `deploy_sample_endpoint.ipynb`: a Jupyter notebook to deploy an the endpoint. You will need a `model.tar.gz`: the parameters for our trained model so we don’t need to train one
- `model.tar.gz`: the parameters for our trained model so we don’t need to train one
- Five Lambda functions
  - `triggerStateMachine.zip`
  - `prepare.zip`
  - `getEndpoints.zip`
  - `callEndpoint.zip`
  - `labelAndSave.zip`
- Three Lambda layers. You will need to [create three layers](https://docs.aws.amazon.com/lambda/latest/dg/creating-deleting-layers.html#layers-create) for the following libraries: [fonts](https://pypi.org/project/fonts/), [pillow](https://pypi.org/project/pillow/), and [opencv-headless](https://pypi.org/project/opencv-python-headless/). Make sure to use the following names:
  - `FontLayer.zip`
  - `OpenCVHeadlessLayer.zip`
  - `PillowLayer.zip`
- `TestImage.zip`: A folder with some images to test the architecture

1. **Create an S3 bucket**

   - Navigate to Amazon S3, select buckets, and select **Create bucket**.
   - Name it anything you want (must be globally unique). For example, resources-XXXX-XXXX-XXXX using your account number.
   - Leave all the other parameters as default.
   - Select **Create bucket**.

2. **Upload all the provided documents to your resources bucket**
   - Once your bucket is deployed, navigate to it and select **Upload**.
   - Drag all the provided files of select **Add** files and locate them locally.
     - NOTE: The files should NOT be in a folder. They should be uploaded as a “flat” hierarchy.
   - Leave everything else as default and select **Upload**. The larger files may take a minute or two to upload depending on your internet connection.

### Deploy the object detection sample endpoint with SageMaker

For our sample model, we deploy an already trained model to recognize bees from images.

1. **Provision a SageMaker Notebook Instance**
   - From your [AWS Management Console](https://aws.amazon.com/console/), navigate to **SageMaker**.
   - In the left panel go to **Notebook > Notebook Instances**.
   - Select **Create notebook instance** in the top-right corner.
   - In Notebook instance settings
     - Name your instance (e.g., _sample-endpoint_).
     - Leave the instance type and platform identifiers as default.
   - In Permissions and encryption
     - Select **Create a new role** in the dropdown and then **Any S3 bucket** in the popup window. Select Create role.
     - Leave the other fields as default.
   - Leave all the optional sections as default and select **Create notebook instance**.
   - Wait 2-3 minutes for the notebook’s status to go from Pending to InService, then select **Open JupyterLab**.
2. **Run the provided notebook**
   - Inside JupyterLab, select the _upload files_ icon. It is the upward-pointing arrow at the top of the left panel.
   - Locate the _deploy_sample_endpoint.ipynb_ in your local machine and open it. Once it appears in the left panel, double-click the file to open the notebook.
   - From the dropdown, select **conda_tensorflow2_p310** for the Kernel and select **Select**.
   - Note that you need to copy the Amazon S3 URI of the _model.tar.gz_ that you uploaded earlier and paste it into the model_artifact line in the second cell.
   - Run the first four cells of notebook but do not run the last cell so you DON’T DELETE THE ENDPOINT.
     - Note that, depending on your [AWS Region](https://aws.amazon.com/about-aws/global-infrastructure/regions_az/), the ml.m5.xlarge instance may not be available. If that's the case and you get an error, then try a similar instance. For example, ml.g5.2xlarge.
   - Back in the Console, navigate back to SageMaker and in the left panel select **Inference > Endpoints** to see the endpoint being deployed. Wait until the endpoint status goes from Creating to _InService_.
   - Copy the _Name_ (not the ARN) of the endpoint and save it somewhere or continue the next steps on a separate window so you have the name available.

### Launch the CloudFormation template to deploy the architecture

1. In the Console, navigate to CloudFormation.
2. Select **Create stack**.
3. Select **Template is ready** and for the source choose **Amazon S3 URL**.
4. Locate the _template.yaml_ file in the resources bucket and copy its URL (not the URI), then paste it into the Amazon S3 URL field.
5. Select **Next**.
6. In _Specify stack details_
   - Provide a name for your stack, such as _ComputerVisionStack_.
   - **EndpointConfigValue**: Paste your endpoint name inside the appropriate place in the EndpointConfigValue parameter. It looks like this:
     {… "ep_name": "tf2-object-detection-xxxx-xx-xx-xx-xx-xx-xxx"…}
   - **ResourcesPublicBucketName**: This is the name of your resource bucket. Update it.
   - **SourceBucketName**: This is the name of the source bucket that CloudFormation creates, so it must be a unique name. We upload the images that we want labeled to this bucket.
   - **TableName**: The name for the DynamoDB table for the model results. You can leave the default name or provide another one.
7. Select **Next**.
8. For stack options, leave everything default and select **Next**.
9. In Review, acknowledge and select **Submit**.
10. Wait until the stack creates.

### Test the solution

1. Navigate to Amazon S3.
2. Find and navigate to the newly created source bucket. It is the source bucket you named in the previous step.
3. Select **Create folder** and call it _raw_images_.
4. Select **Create folder**.
5. In the _raw_images_ folder, upload one of the test images (or multiple if you prefer). After a few seconds, you should see two folders created in the bucket: _resized_images/_ and _labeled_images/_ with the corresponding results inside.
6. You can examine the endpoint output in the results table by navigating to _DynamoDB > Tables > Explore items_ and selecting the results table.
7. You can examine the state of the state machine by navigating to _Step Functions > State machines > ComputerVisionOrchestrator_.

Don’t forget to [delete the stack](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-delete-stack.html) and SageMaker resources after you are done so that you don’t incur unwanted charges. For SageMaker you just need to finish running the last cell to delete the endpoint and then delete the notebook instance.

Congratulations on deploying the solution!

# SAMPLE SAMPLE SAMPLE SAMPLE SAMPLE SAMPLE SAMPLE SAMPLE SAMPLE SAMPLE SAMPLE SAMPLE

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.aws.dev/vshadang/remote-communications-infrastructure-with-computer-vision.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.aws.dev/vshadang/remote-communications-infrastructure-with-computer-vision/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

---

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name

Choose a self-explaining name for your project.

## Description

Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges

On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals

Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation

Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage

Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support

Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap

If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing

State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment

Show your appreciation to those who have contributed to the project.

## License

For open source projects, say how it is licensed.

## Project status

If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
