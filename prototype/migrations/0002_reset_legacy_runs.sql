-- Method IDs were renumbered to the seven-method scheme (M1..M7): the offline
-- ONNX pack is now two explicit methods (M2 single, M3 always 2x2 tiled) and
-- every cloud method shifted by one. Rows written under the old numbering
-- cannot be attributed to a method unambiguously, so the run history is reset.
DELETE FROM run_methods;
DELETE FROM runs;
