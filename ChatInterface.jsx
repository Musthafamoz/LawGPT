import React, { useState } from 'react';
import './style.css';
import loadingGif from '../assets/Loading_2.gif'; // Replace with your correct loading GIF path
import wait from '../assets/inf.gif';

function Chatbot() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);

  const handleAskQuestion = async () => {
    if (question.trim() === '') return; // Prevent sending empty questions
    setLoading(true);

    // Add the user question to chat history
    setChatHistory((prevHistory) => [...prevHistory, { type: 'user', text: question }]);

    try {
      const response = await fetch('http://localhost:5000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });
      const data = await response.json();

      // Once the response is received, add the bot's message to chat history
      setChatHistory((prevHistory) => [
        ...prevHistory, // Keep previous messages
        { type: 'bot', text: data.answer || 'No answer found.' },
      ]);
    } catch (error) {
      console.error('Error fetching answer:', error);
      
      // Add an error message in case of failure
      setChatHistory((prevHistory) => [
        ...prevHistory,
        { type: 'bot', text: 'Error fetching answer' },
      ]);
    } finally {
      setLoading(false);
      setQuestion(''); // Clear the input field after asking
    }
  };

  return (
    <div className="main">
      <div className="chat-container">
        {chatHistory.map((message, index) => (
          <div
            key={index}
            className={`message ${message.type === 'user' ? 'user-message' : 'bot-message'}`}
          >
            <p>{message.text}</p>
          </div>
        ))}

        {/* Display the loading indicator if the bot is still processing */}
        {loading && (
          <div className="loading-container">
            <img src={wait} alt="Loading..." className="loading-indicator" />
          </div>
        )}
      </div>

      <div className="Question">
        <input
          className="qinput"
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Enter your question"
        />
        
        {loading ? (
          <img src={loadingGif} alt="Computer man" style={{ width: '48px', height: '48px' }} />
        ) : (
          <button onClick={handleAskQuestion} disabled={loading}>Ask</button>
        )}
      </div>
    </div>
  );
}

export default Chatbot;
