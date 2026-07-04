using Microsoft.AspNetCore.Mvc;

[ApiController]
[Route("api/mission")]
public class MissionControlController : ControllerBase
{
    [HttpPost("takeoff")]
    public async Task<IActionResult> TriggerSwarmTakeoff([FromQuery] int altitude)
    {
        // 1. Construct parameters
        var paramsDict = new Dictionary<string, string>
        {
            { "altitude", altitude.ToString() }
        };

        // 2. Broadcast to all Python clients via the gRPC static helper
        await DroneHubService.BroadcastCommandAsync("TAKEOFF", paramsDict);

        return Ok($"Takeoff signal sent to swarm. Target Altitude: {altitude}m");
    }

    [HttpPost("scan")]
    public async Task<IActionResult> StartScanning()
    {
         await DroneHubService.BroadcastCommandAsync("SCAN_TARGET", new Dictionary<string, string>{ {"mode", "thermal"} });
         return Ok("Swarm is now scanning.");
    }
}