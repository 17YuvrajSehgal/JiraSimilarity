CREATE OR REPLACE VIEW tawos_issue_search_document AS
SELECT
    i.ID AS issue_id,
    i.Issue_Key AS issue_key,
    p.Project_Key AS project_key,
    i.Title AS title,
    COALESCE(i.Description_Text, i.Description, '') AS description_text,
    i.Type AS issue_type,
    i.Priority AS priority,
    i.Status AS status,
    i.Resolution AS resolution,
    COALESCE(component_data.components, '') AS components,
    COALESCE(affected_data.affected_versions, '') AS affected_versions,
    COALESCE(fix_data.fix_versions, '') AS fix_versions,
    COALESCE(link_data.linked_issue_ids, '') AS linked_issue_ids,
    COALESCE(link_data.duplicate_issue_ids, '') AS duplicate_issue_ids
FROM Issue i
JOIN Project p ON p.ID = i.Project_ID
LEFT JOIN (
    SELECT
        ic.Issue_ID,
        GROUP_CONCAT(DISTINCT c.Name ORDER BY c.Name SEPARATOR '||') AS components
    FROM Issue_Component ic
    JOIN Component c ON c.ID = ic.Component_ID
    GROUP BY ic.Issue_ID
) AS component_data ON component_data.Issue_ID = i.ID
LEFT JOIN (
    SELECT
        av.Issue_ID,
        GROUP_CONCAT(DISTINCT v.Name ORDER BY v.Name SEPARATOR '||') AS affected_versions
    FROM Affected_Version av
    JOIN Version v ON v.ID = av.Affected_Version_ID
    GROUP BY av.Issue_ID
) AS affected_data ON affected_data.Issue_ID = i.ID
LEFT JOIN (
    SELECT
        fv.Issue_ID,
        GROUP_CONCAT(DISTINCT v.Name ORDER BY v.Name SEPARATOR '||') AS fix_versions
    FROM Fix_Version fv
    JOIN Version v ON v.ID = fv.Fix_Version_ID
    GROUP BY fv.Issue_ID
) AS fix_data ON fix_data.Issue_ID = i.ID
LEFT JOIN (
    SELECT
        Issue_ID,
        GROUP_CONCAT(DISTINCT Target_Issue_ID ORDER BY Target_Issue_ID SEPARATOR '||') AS linked_issue_ids,
        GROUP_CONCAT(
            DISTINCT CASE
                WHEN LOWER(Name) = 'duplicate' THEN Target_Issue_ID
                ELSE NULL
            END
            ORDER BY Target_Issue_ID SEPARATOR '||'
        ) AS duplicate_issue_ids
    FROM Issue_Link
    GROUP BY Issue_ID
) AS link_data ON link_data.Issue_ID = i.ID;

