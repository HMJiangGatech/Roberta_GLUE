# This is the docker file built for GLUE task
FROM tscience.azurecr.io/biglm/biglm:1.12-0.4.1-cuda9.2-py36

# vim setup
RUN git clone --depth=1 https://github.com/amix/vimrc.git /root/.vim_runtime
RUN sh /root/.vim_runtime/install_awesome_vimrc.sh
RUN git clone https://github.com/Xuyuanp/nerdtree-git-plugin.git /root/.vim_runtime/my_plugins/nerdtree-git-plugin
COPY my_configs.vim /root/.vim_runtime/my_configs.vim

RUN pip3 install torch torchvision

RUN git config --global user.name "v-hajia"
RUN git config --global user.email "jianghm.ustc@gmail.com"

WORKDIR /work

