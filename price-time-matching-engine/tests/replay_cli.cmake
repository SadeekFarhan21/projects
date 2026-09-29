# End-to-end check of the CLI: generate a log, replay it to an output log,
# then verify that a second, independent replay matches line for line.
file(MAKE_DIRECTORY ${WORK})
execute_process(COMMAND ${EXCH} gen --n 20000 --seed 42 --out ${WORK}/in.log RESULT_VARIABLE r1)
if(NOT r1 EQUAL 0)
  message(FATAL_ERROR "gen failed")
endif()
execute_process(COMMAND ${EXCH} replay --in ${WORK}/in.log --out ${WORK}/out.log RESULT_VARIABLE r2
                OUTPUT_VARIABLE replay_out)
if(NOT r2 EQUAL 0)
  message(FATAL_ERROR "replay failed")
endif()
string(REGEX MATCH "digest +([0-9a-f]+)" _ "${replay_out}")
set(digest ${CMAKE_MATCH_1})
execute_process(COMMAND ${EXCH} verify --in ${WORK}/in.log --expect ${WORK}/out.log --digest ${digest}
                RESULT_VARIABLE r3 OUTPUT_VARIABLE verify_out)
message(STATUS "${verify_out}")
if(NOT r3 EQUAL 0)
  message(FATAL_ERROR "verify failed")
endif()
