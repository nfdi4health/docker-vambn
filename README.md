# docker-vambn

A containerized implementation of the VAMBN approach by TA6.4.

This version does not contain ADNI data. The execution guide below acts as a general reference for executing your own data with VAMBN.

############## VAMBN Execution Guide ################################

1. Please make sure that:
   - You have permission to use the ADNI data.
   - You have root or admin access to your machine.
   - You have installed Git and Docker on your machine.
     (Install instruction reference for Windows: (Git) https://git-scm.com/downloads, (Docker) https://docs.docker.com/desktop/windows/install/)

2. Git clone this repo at your local folder by executing the following command your terminal:
   $ git clone https://gitlab.scai.fraunhofer.de/hwei.geok.ng/docker-vambn.git

3. From the terminal, change the directory to the project folder.
   $ cd (your directory)/docker-vambn

4. Create a docker using the Dockerfile in the folder by executing the following command:
   $ docker build -t vambn-adni .
   Please remember to include the "dot" at the end of "vambn-adni", it's not a full stop, it indicates the current directory.

5. Once the vambn-adni docker is created, connect your local data with the docker data and run the docker in interactive mode (please replace the directory according to your folder names):
   $ docker run -it -v C:/Users/(your directory)/docker-vambn-adni/vambn/01_data:/vambn/01_data -v C:/Users/(your directory)/docker-vambn-adni/vambn/04_output:/vambn/04_output vambn-adni

6. Now, we are ready to execute the VAMBN code. Execute the following command directly:
   $ Rscript vambn/03_code/main.R 0

7. Let the code run for a few hours. This might result in two scenarios: either the code can be executed successfully till the very end, or you will encounter an error message.

8. To exit the docker interactive mode, execute the following command:
   $ exit


############## End of VAMBN Execution Guide ##############################
