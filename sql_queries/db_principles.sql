-- Detailed view of all principals with their identifiers
SELECT 
    p.principal_id,
    p.name,
    p.type,
    p.type_desc,
    p.default_schema_name,
    p.create_date,
    p.modify_date,
    p.is_fixed_role,
    CONVERT(varchar(85), p.sid, 1) AS sid_hex,
    LEN(p.sid) AS sid_length,
    -- Role memberships
    STRING_AGG(r.name, ', ') AS roles
FROM sys.database_principals p
LEFT JOIN sys.database_role_members rm ON p.principal_id = rm.member_principal_id
LEFT JOIN sys.database_principals r ON rm.role_principal_id = r.principal_id AND r.type = 'R'
WHERE p.type IN ('S', 'U', 'G', 'E', 'R')
    AND p.name NOT IN ('public')
GROUP BY p.principal_id, p.name, p.type, p.type_desc, p.default_schema_name, 
         p.create_date, p.modify_date, p.is_fixed_role, p.sid
ORDER BY p.type_desc, p.name;