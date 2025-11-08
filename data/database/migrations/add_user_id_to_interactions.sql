-- Migration: Add user_id column to user_interactions table
-- Date: 2024-11-09
-- Purpose: Enable proper preference tracking across sessions by user_id

-- Add user_id column if it doesn't exist
ALTER TABLE user_interactions 
ADD COLUMN IF NOT EXISTS user_id VARCHAR(100);

-- Create index on user_id for query performance
CREATE INDEX IF NOT EXISTS idx_interactions_user ON user_interactions(user_id);

-- Optional: Backfill user_id from session_id for existing records
-- (Only if session_id was being used as user_id previously)
-- UPDATE user_interactions SET user_id = session_id WHERE user_id IS NULL;

-- Add comment
COMMENT ON COLUMN user_interactions.user_id IS 'User identifier for preference learning across sessions';

