cd ../../../
set PROTOC=3rdparty/tools/protoc-2.6.1.exe

echo caffe.pb.h is being generated
"%PROTOC%" -I="src/caffe/proto" --cpp_out="src/caffe/proto/" "src/caffe/proto/caffe.proto"

copy /y "src\\caffe\\proto\\caffe.pb.h" "include\\caffe\\proto\\caffe.pb.h"

REM echo caffe_pretty_print.pb.h is being generated
REM "%PROTOC%" -I="src/caffe/proto" --cpp_out="src/caffe/proto/" "src/caffe/proto/caffe_pretty_print.proto"

REM copy /y "src\\caffe\\proto\\caffe_pretty_print.pb.h" "proto\\caffe_pretty_print.pb.h"

pause

