#!/bin/bash

if [[ $# -ne 1 ]]; then
  echo "Run as following:"
  echo "stage_checkpoint.sh <task_name>"
  exit 1
fi
DIR=checkpoints
TASKS=$1
if [ "$TASKS" = "ALL" ]
then
  TASKS="QQP MNLI QNLI MRPC RTE STS-B SST-2 CoLA WSC"
fi

while true; do
    read -p "Stage ALL ? [Y/n]" yn
    case $yn in
        [Y]* ) NOT_STAGEALL=false; echo "Stage ALL checkpoints that is unstaged"; break;;
        [n]* ) NOT_STAGEALL=true; echo "Manual Decide Staging"; break;;
        * ) echo "Please answer Y or n.";;
    esac
done

for TASK in $TASKS
do
    SUBDIR="$DIR/$TASK"
    echo "Staging $SUBDIR"
    for CKPDIR in $SUBDIR/*/
    do
        if [[ $CKPDIR != *_staged/ ]]; then
            yn=n
            while $NOT_STAGEALL ; do
                read -p "Stage $CKPDIR ? [Y/n]" yn
                case $yn in
                    [Y]* ) break;;
                    [n]* ) break;;
                    *) echo "Please answer Y or n.";;
                esac
            done
            if [[ $yn == Y  ]] || [[ $NOT_STAGEALL == false  ]]
            then
                NEW_CKPDIR="${CKPDIR%/}_staged"
                if [ -d $NEW_CKPDIR ]
                then
                    while true ; do
                        read -p "$NEW_CKPDIR exists. Still stage $CKPDIR ? [Y/n]" yn
                        case $yn in
                            [Y]* )
                                echo "stage $CKPDIR to $NEW_CKPDIR"
                                rm -rf $NEW_CKPDIR
                                mv $CKPDIR $NEW_CKPDIR
                                break;;
                            [n]* ) break;;
                            *) echo "Please answer Y or n.";;
                        esac
                    done
                else
                    echo "stage $CKPDIR to $NEW_CKPDIR"
                    mv $CKPDIR $NEW_CKPDIR
                fi
            fi
        fi
    done
done
