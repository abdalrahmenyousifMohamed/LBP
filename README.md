# LBP
 To add repository and tag for the docker, we run
 
 ` docker build --compress -t ml/project1:latest .
`

 in the project root folder.

` docker run -p 8080:8000  <CONTAINER_ID> or NAME `

After testing, we can run ` docker push ` command to push our image to docker image registry of cloud providers (such as AWS ECR and GCP Container Registry). You’ll need to do authentication before being able to push to the repository. Follow the guides from different cloud providers, which will vary from using simple ` docker login ` to installing CLI tool. The command should look like this:

` docker tag ml/project1 [SOME_CONTAINER_REGISTRY]/ml/project1:latest `

` docker push [SOME_CONTAINER_REGISTRY]/ml/project1:latest `
