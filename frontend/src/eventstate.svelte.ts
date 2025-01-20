// This file contains the interfaces, definition for all the event handler in application
// #Note: This is global event handler implementation

// type decleartion for the registerable listeners
type generic_callbackfn = (...args : any) => void | {};

// type for the Global Event map
// identifier (i.e Event name), listeners for that event name
// #Note: name conflict will result in unwanted behavious so make sure the event name are unique all across
// eg: { "Generate", [listener1, listener2] }
// This will register new event name "Generate" in event map (if doesn't exists) and append the listener1, listener2 in "Generate" pair
// When DisptachListenerIn Method is called with "Generate" param, all the listener will be invoked (with params if any)
type IEventMap = Map<string, generic_callbackfn[]>;

// add new or append listener to given event name
// method is expected to be called at `onMount` of components 
export const AddListenerIn = (identifier: string, subscriber : generic_callbackfn) => {
    console.log(identifier)
    
    if(EventMap?.has(identifier))
    {
        console.log("Identifier Already exists so only adding to listener");
    }
    else 
    {
        console.log("Identifier doesn't exist creating new and adding listener");
        EventMap?.set(identifier, []);
    }
    EventMap?.get(identifier)?.push(subscriber);
    // PrintEventMap()
}

// call with name of event to dispatch listeners for
export const DispatchListenerIn = (identifier : string, ...args : any) => {
    if(EventMap?.has(identifier))
    {
        EventMap.get(identifier)?.forEach((callbacks) => {
            if(callbacks)   callbacks(...args)
        })
    }
}

export function PrintEventMap()
{
    if(EventMap.size == 0)
    {
        console.log("No entry in map")
    }
    EventMap?.forEach((lis, key) => {
        console.log("Key : " + key + " || Listener Count : ", lis.length);
    })
}

// call in onMount function of main entry point to reset in every page refresh
export function InitEventMap()
{
    EventMap.clear();
}

export let EventMap = $state<IEventMap>(new Map());