using Grpc.Core;
using System.Collections.Concurrent;
using SwarmBridge; // Generated from your .proto file

public class DroneHubService : DroneHub.DroneHubBase
{
    // Thread-safe dictionary to store active drone connections
    // Key: DroneID, Value: The output stream to send commands to that drone
    private static readonly ConcurrentDictionary<string, IServerStreamWriter<HiveCommand>> _activeDrones = new();

    public override async Task ConnectToHive(
        DroneIdentity request, 
        IServerStreamWriter<HiveCommand> responseStream, 
        ServerCallContext context)
    {
        string droneId = request.DroneId;
        Console.WriteLine($"[Hive] Drone Connected: {droneId} (Group: {request.SwarmGroup})");

        // 1. Register the drone (Save its stream so we can talk to it later)
        _activeDrones.TryAdd(droneId, responseStream);

        try
        {
            // 2. Keep the connection alive indefinitely.
            // We wait until the client disconnects or the token is cancelled.
            while (!context.CancellationToken.IsCancellationRequested)
            {
                // In a real scenario, you might ping/pong here to check health
                await Task.Delay(1000); 
            }
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine($"[Hive] Connection closed for {droneId}");
        }
        finally
        {
            // 3. Cleanup when drone disconnects
            _activeDrones.TryRemove(droneId, out _);
            Console.WriteLine($"[Hive] Drone Disconnected: {droneId}");
        }
    }

    // A helper method for your Backend Logic to call
    public static async Task BroadcastCommandAsync(string commandName, Dictionary<string, string> parameters)
    {
        var commandPacket = new HiveCommand { CommandName = commandName };
        if (parameters != null)
        {
            foreach (var p in parameters) commandPacket.Parameters.Add(p.Key, p.Value);
        }

        Console.WriteLine($"[Hive] Broadcasting {commandName} to {_activeDrones.Count} drones...");

        // Send to all connected drones in parallel
        foreach (var drone in _activeDrones)
        {
            try
            {
                await drone.Value.WriteAsync(commandPacket);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to send to {drone.Key}: {ex.Message}");
            }
        }
    }
}