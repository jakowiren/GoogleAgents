from google.adk.tools.tool_context import ToolContext
from typing import Optional, Dict, Any

def block_paris_tool_guardrail(
    tool: 'BaseTool', 
    args: Dict[str, Any], 
    tool_context: ToolContext
) -> Optional[Dict]:
    """Blocks weather requests for Paris."""
    tool_name = tool.name
    agent_name = tool_context.agent_name
    print(f"--- Callback: block_paris_tool_guardrail running for tool '{tool_name}' in agent '{agent_name}' ---")
    print(f"--- Callback: Inspecting args: {args} ---")

    # Check if it's a weather request for Paris
    if tool_name == "get_weather":
        city = args.get("city", "").lower()
        if city == "paris":
            print(f"--- Callback: Blocking weather request for Paris ---")
            tool_context.state["tool_block_triggered"] = True
            return {
                "status": "error",
                "error_message": "Weather information for Paris is currently unavailable due to policy restrictions."
            }
    
    print(f"--- Callback: Allowing tool '{tool_name}' to proceed ---")
    return None

def get_weather(city: str, tool_context: ToolContext) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").
        context: The invocation context containing session state.

    Returns:
        dict: A dictionary containing the weather information.
    """
    print(f"--- Tool: get_weather called for city: {city} ---")
    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius")
    print(f"--- Tool: Reading state 'user_preference_temperature_unit': {preferred_unit} ---")
    city_normalized = city.lower().replace(" ", "")

    # Mock weather data (stored in Celsius)
    mock_weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
        "paris": {"temp_c": 20, "condition": "partly cloudy"},  # Added but will be blocked
    }

    if city_normalized in mock_weather_db:
        data = mock_weather_db[city_normalized]
        temp_c = data["temp_c"]
        condition = data["condition"]

        # Format temperature based on preference
        if preferred_unit == "Fahrenheit":
            temp_value = (temp_c * 9/5) + 32
            temp_unit = "°F"
        else:
            temp_value = temp_c
            temp_unit = "°C"

        report = f"The weather in {city.capitalize()} is {condition} with a temperature of {temp_value:.0f}{temp_unit}."
        return {"status": "success", "report": report}
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I don't have timezone information for {city}."
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    return {"status": "success", "report": report}

def say_hello(name: str = "there") -> str:
    """Provides a simple greeting, optionally addressing the user by name.

    Args:
        name (str, optional): The name of the person to greet. Defaults to "there".

    Returns:
        str: A friendly greeting message.
    """
    print(f"--- Tool: say_hello called with name: {name} ---")
    return f"Hello, {name}!"

def say_goodbye() -> str:
    """Provides a simple farewell message to conclude the conversation."""
    print(f"--- Tool: say_goodbye called ---")
    return "Goodbye! Have a great day."
