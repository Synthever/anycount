-- Records whether a run counted on the RGB-negative derivative of the image.
-- The derivative itself lives under {data_dir}/derived/ so repeated runs reuse it.
ALTER TABLE runs ADD COLUMN inverted INTEGER NOT NULL DEFAULT 0;
