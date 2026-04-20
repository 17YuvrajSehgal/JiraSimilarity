SELECT DISTINCT
    LEAST(Issue_ID, Target_Issue_ID) AS left_issue_id,
    GREATEST(Issue_ID, Target_Issue_ID) AS right_issue_id
FROM Issue_Link
WHERE LOWER(Name) = 'duplicate'
  AND Issue_ID IS NOT NULL
  AND Target_Issue_ID IS NOT NULL;

