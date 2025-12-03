-- this can be an entra user like XXXXX@domain.com or it can be the name of the app service's managed identity
CREATE USER [<username>] FROM EXTERNAL PROVIDER;

-- Grant necessary permissions (adjust based on your needs)
ALTER ROLE db_datareader ADD MEMBER [<username>];
